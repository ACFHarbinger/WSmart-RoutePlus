"""Vectorised HVPL policy: the simulator's Hybrid VPL (``HVPLConfig``) on a batch.

Mirrors :class:`~logic.src.policies.routing.meta_heuristics.hybrid_volleyball_premier_league.solver.HVPLSolver`
(Sun et al. 2023) with every team of every instance processed together:

1. hyper-heuristic ACO initialisation (``HyperHeuristicACO.construct``, run per row with
   the simulator class), twice (active and passive leagues): ``aco_init_iterations``
   constructions, padded with ``build_greedy_routes`` solutions up to ``n_teams``, the
   ``n_teams`` most profitable kept;
2. ``max_iterations`` seasons (or ``time_limit`` seconds), each with
   - HGS evolution: ``n_teams`` offspring from size-3 tournaments, crossover with
     probability ``crossover_rate`` (union of the parents' bins, cut to the
     ``int(1.1 * |parent1|)`` heaviest when larger, rebuilt by greedy insertion) and
     mutation with probability ``mutation_rate`` (random removal of
     ``max(2, int(0.2 n))`` bins + greedy reinsertion);
   - selection of the ``n_teams`` best of parents and offspring;
   - substitution of the ``max(1, int(n_teams * substitution_rate))`` weakest teams by
     random passive teams;
   - ALNS coaching of every team with ``alns_params``;
   - best tracking (the simulator's pheromone deposit on a new best is never read
     after initialisation, so it has no effect and is not mirrored).

Attributes:
    VectorizedHVPL: Batched HVPL expert configured by ``HVPLConfig``.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional

import torch

from logic.src.configs.policies.routing.meta_heuristics.hvpl import HVPLConfig
from logic.src.policies.vectorized.routing.shared import routes as rs
from logic.src.policies.vectorized.routing.shared.base import VectorizedRoutingPolicy
from logic.src.policies.vectorized.routing.shared.vrpp import VRPPBatch

from .adaptive_large_neighborhood_search import VectorizedALNSEngine


class VectorizedHVPL(VectorizedRoutingPolicy):
    """Batched counterpart of the simulator's ``HVPLPolicy`` / ``HVPLSolver``.

    Takes exactly the ``HVPLConfig`` fields (``aco``/``aco_params``, ``alns``/``alns_params``);
    ``packing`` and ``route_improvement`` are simulator-only and ignored.
    """

    config_cls = HVPLConfig

    # ------------------------------------------------------------------
    def _rep(self, inst: VRPPBatch, T: int) -> VRPPBatch:
        return inst.subset(torch.arange(inst.batch_size, device=inst.device).repeat_interleave(T))

    def _greedy_teams(self, inst: VRPPBatch, T: int) -> torch.Tensor:
        L = rs.width(inst.n_bins)
        return rs.greedy_profit_construction(self._rep(inst, T), self.generator).view(inst.batch_size, T, L)

    def _hh_aco_solvers(self, inst: VRPPBatch) -> list:
        """One simulator ``HyperHeuristicACO`` per row, as ``HVPLSolver.aco_solver``.

        The hyper-heuristic ACO builds solutions by bootstrapping with a uniformly drawn
        ``recreate_repair`` heuristic and applying a roulette-selected operator sequence;
        its eleven insertion heuristics (regret-k, GENI, LDS branch-and-bound, ...) and
        eleven LS/ruin operators have no batched form, so construction runs row by row
        with the simulator's own class, its ``seed`` drawn from ``self.generator``. The
        node-level pheromones HVPL deposits are never read by ``construct`` (and HVPL
        constructs only before its first deposit), so they are not mirrored.
        """
        from logic.src.policies.routing.base.sub_problem import build_subproblem
        from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_aco import (
            HyperHeuristicACO,
        )

        dist = inst.dist.detach().cpu().double().numpy()
        waste = inst.waste.detach().cpu().double().numpy()
        cap = inst.capacity.detach().cpu().double().tolist()
        seeds = rs._row_seeds(inst.batch_size, self.generator, inst.device)
        solvers = []
        for b in range(inst.batch_size):
            aco_cfg = dataclasses.replace(self.cfg.aco_params, seed=int(seeds[b]) % (2**32))
            problem = build_subproblem(
                dist_matrix=dist[b],
                wastes={i: float(waste[b, i]) for i in range(1, inst.n_bins + 1)},
                capacity=float(cap[b]),
                revenue=float(inst.R),
                cost_unit=float(inst.C),
                mandatory_nodes=[],
            )
            solvers.append(HyperHeuristicACO(problem, aco_cfg))
        return solvers

    def _league(self, inst: VRPPBatch, solvers: list) -> torch.Tensor:
        """``HVPLSolver._aco_initialization``: HH-ACO constructions, greedy fill, top ``n_teams``."""
        cfg = self.cfg
        T = int(cfg.n_teams)
        n_init = int(cfg.aco_init_iterations)
        B, N, L = inst.batch_size, inst.n_bins, rs.width(inst.n_bins)
        nodes = list(range(1, N + 1))
        built = [[r for r in (solvers[b].construct(nodes, []) for _ in range(n_init)) if r] for b in range(B)]
        P = max(T, max((len(x) for x in built), default=0))
        pool = rs.empty_routes(B * P, N, inst.device).view(B, P, L).clone()
        n_fill = []
        for b in range(B):
            if built[b]:
                pool[b, : len(built[b])] = rs.from_route_lists(built[b], N, inst.device)
            n_fill.append(max(0, T - len(built[b])))
        k = max(n_fill)
        if k > 0:
            greedy = self._greedy_teams(inst, k)
            for b in range(B):
                c = len(built[b])
                pool[b, c : c + n_fill[b]] = greedy[b, : n_fill[b]]
        sizes = torch.tensor([max(T, len(x)) for x in built], device=inst.device)
        prof = rs.seq_profit(pool.reshape(-1, L), self._rep(inst, P)).view(B, P)
        prof = torch.where(torch.arange(P, device=inst.device).unsqueeze(0) < sizes.unsqueeze(1), prof, -float("inf"))
        top = torch.argsort(-prof, dim=1, stable=True)[:, :T]
        return torch.gather(pool, 1, top.unsqueeze(2).expand(-1, -1, L))

    def _profits(self, teams: torch.Tensor, inst: VRPPBatch) -> torch.Tensor:
        B, T, L = teams.shape
        return rs.seq_profit(teams.reshape(B * T, L), self._rep(inst, T)).view(B, T)

    def _tournament(self, profits: torch.Tensor, n: int, k: int = 3) -> torch.Tensor:
        B, T = profits.shape
        r = torch.rand(B, n, T, generator=self.generator, device=profits.device)
        picks = torch.argsort(r, dim=2)[:, :, : min(k, T)]
        p = torch.gather(profits.unsqueeze(1).expand(-1, n, -1), 2, picks)
        return torch.gather(picks, 2, p.argmax(dim=2, keepdim=True)).squeeze(2)

    # ------------------------------------------------------------------
    def solve(self, inst: VRPPBatch, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the batched HVPL.

        Args:
            inst: Instances.
            initial: Ignored.

        Returns:
            torch.Tensor: Best route sequences.
        """
        cfg = self.cfg
        alns_cfg = cfg.alns_params
        B, N, M = inst.batch_size, inst.n_bins, inst.n_bins + 1
        dev = inst.device
        if N == 0:
            return rs.empty_routes(B, N, dev)
        start = time.perf_counter()
        T = int(cfg.n_teams)
        solvers = self._hh_aco_solvers(inst)
        active = self._league(inst, solvers)
        passive = self._league(inst, solvers)
        prof = self._profits(active, inst)
        order = torch.argsort(-prof, dim=1, stable=True)
        active = torch.gather(active, 1, order.unsqueeze(2).expand_as(active))
        prof = torch.gather(prof, 1, order)
        best_S, best_p = active[:, 0].clone(), prof[:, 0].clone()
        alns = VectorizedALNSEngine(alns_cfg, self.generator, self.expand_pool)
        L = active.size(2)
        repT = self._rep(inst, T)
        pa = bool(cfg.profit_aware_operators)
        for season in range(int(cfg.max_iterations)):
            if cfg.time_limit > 0 and time.perf_counter() - start > cfg.time_limit:
                break
            # HGS evolution.
            i1 = self._tournament(prof, T)
            i2 = self._tournament(prof, T)
            p1 = torch.gather(active, 1, i1.unsqueeze(2).expand(-1, -1, L)).reshape(B * T, L)
            p2 = torch.gather(active, 1, i2.unsqueeze(2).expand(-1, -1, L)).reshape(B * T, L)
            child = p1.clone()
            do_x = torch.rand(B * T, generator=self.generator, device=dev) < cfg.crossover_rate
            if bool(do_x.any()):
                v1, v2 = rs.visited(p1, M), rs.visited(p2, M)
                union = v1 | v2
                n1 = v1.sum(dim=1)
                keep_n = (n1.float() * 1.1).long()
                prune = union.sum(dim=1) > n1
                w = torch.where(union, repT.waste, torch.full_like(repT.waste, -1.0))
                rank = torch.argsort(torch.argsort(-w, dim=1, stable=True), dim=1)
                pruned = union & (rank < keep_n.unsqueeze(1))
                nodes = torch.where(prune.unsqueeze(1), pruned, union)
                if self.expand_pool:
                    nodes = torch.ones_like(nodes)
                    nodes[:, 0] = False
                rebuilt = rs.greedy_insertion(torch.zeros_like(p1), nodes, repT, pa)
                child = torch.where(do_x.unsqueeze(1), rebuilt, child)
            do_m = torch.rand(B * T, generator=self.generator, device=dev) < cfg.mutation_rate
            if bool(do_m.any()):
                n_rm = torch.clamp((rs.n_visited(child).float() * 0.2).long(), min=2)
                Pm, pos = rs.random_removal(child, n_rm, self.generator)
                removed = torch.zeros(B * T, M, dtype=torch.bool, device=dev)
                removed.scatter_(1, torch.where(pos, child, torch.zeros_like(child)), True)
                removed[:, 0] = False
                pending = removed | (~rs.visited(Pm, M) if self.expand_pool else removed)
                pending[:, 0] = False
                mutated = rs.greedy_insertion(Pm, pending, repT, pa)
                child = torch.where(do_m.unsqueeze(1), mutated, child)
            off = child.view(B, T, L)
            comb = torch.cat([active, off], dim=1)
            comb_p = torch.cat([prof, self._profits(off, inst)], dim=1)
            top = torch.argsort(-comb_p, dim=1, stable=True)[:, :T]
            active = torch.gather(comb, 1, top.unsqueeze(2).expand(-1, -1, L))
            # Substitution of the weakest teams by random passive teams.
            n_sub = max(1, int(T * cfg.substitution_rate))
            pick = torch.randint(0, passive.size(1), (B, n_sub), generator=self.generator, device=dev)
            active[:, T - n_sub :] = torch.gather(passive, 1, pick.unsqueeze(2).expand(-1, -1, L))
            # ALNS coaching of every team.
            coached, _ = alns.solve(repT, active.reshape(B * T, L))
            active = coached.view(B, T, L)
            prof = self._profits(active, inst)
            order = torch.argsort(-prof, dim=1, stable=True)
            active = torch.gather(active, 1, order.unsqueeze(2).expand_as(active))
            prof = torch.gather(prof, 1, order)
            imp = prof[:, 0] > best_p
            best_S = torch.where(imp.unsqueeze(1), active[:, 0], best_S)
            best_p = torch.where(imp, prof[:, 0], best_p)
            self._viz_record(iteration=season, best_profit=float(best_p.mean()))
        return best_S
