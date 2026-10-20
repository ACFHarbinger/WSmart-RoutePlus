"""Vectorised HVPL policy: the simulator's Hybrid VPL (``HVPLConfig``) on a batch.

Mirrors :class:`~logic.src.policies.routing.meta_heuristics.hybrid_volleyball_premier_league.solver.HVPLSolver`
(Sun et al. 2023) with every team of every instance processed together:

1. ACO initialisation, twice (active and passive leagues): ``aco_init_iterations`` ant
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
   - best tracking; on a new best its edges get ``1 / cost`` pheromone.

Attributes:
    VectorizedHVPL: Batched HVPL expert configured by ``HVPLConfig``.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from logic.src.configs.policies.hvpl import HVPLConfig
from logic.src.policies.vector.shared import routes as rs
from logic.src.policies.vector.shared.base import VectorizedRoutingPolicy
from logic.src.policies.vector.shared.vrpp import VRPPBatch

from .adaptive_large_neighborhood_search import VectorizedALNSEngine
from .ant_colony_system import candidate_mask, construct_ants


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

    def _league(self, inst: VRPPBatch, tau: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        aco = cfg.aco_params
        T = int(cfg.n_teams)
        n_init = int(cfg.aco_init_iterations)
        parts = []
        if n_init > 0:
            parts.append(
                construct_ants(
                    inst,
                    tau,
                    cand,
                    n_init,
                    aco.alpha,
                    aco.beta,
                    bool(cfg.profit_aware_operators),
                    self.generator,
                    self.expand_pool,
                )
            )
        if n_init < T:
            parts.append(self._greedy_teams(inst, T - n_init))
        pool = torch.cat(parts, dim=1)
        if pool.size(1) <= T:
            return pool
        P = pool.size(1)
        prof = rs.seq_profit(pool.reshape(-1, pool.size(2)), self._rep(inst, P)).view(inst.batch_size, P)
        top = torch.argsort(-prof, dim=1, stable=True)[:, :T]
        return torch.gather(pool, 1, top.unsqueeze(2).expand(-1, -1, pool.size(2)))

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
        aco, alns_cfg = cfg.aco_params, cfg.alns_params
        B, N, M = inst.batch_size, inst.n_bins, inst.n_bins + 1
        dev = inst.device
        if N == 0:
            return rs.empty_routes(B, N, dev)
        start = time.perf_counter()
        T = int(cfg.n_teams)
        tau0 = float(aco.tau_0) if aco.tau_0 is not None else 1.0
        tau = torch.full((B, M, M), tau0, device=dev)
        cand = candidate_mask(inst, int(getattr(cfg.aco, "k_sparse", 10)))
        active = self._league(inst, tau, cand)
        passive = self._league(inst, tau, cand)
        prof = self._profits(active, inst)
        order = torch.argsort(-prof, dim=1, stable=True)
        active = torch.gather(active, 1, order.unsqueeze(2).expand_as(active))
        prof = torch.gather(prof, 1, order)
        best_S, best_p = active[:, 0].clone(), prof[:, 0].clone()
        alns = VectorizedALNSEngine(alns_cfg, self.generator, self.expand_pool)
        b = torch.arange(B, device=dev)
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
            if bool(imp.any()):
                cost = rs.seq_cost(best_S, inst).clamp(min=1e-12)
                u, v = best_S[:, :-1], best_S[:, 1:]
                edge = ((u != 0) | (v != 0)) & imp.unsqueeze(1)
                delta = (1.0 / cost).unsqueeze(1).expand_as(u)
                tau.index_put_((b.unsqueeze(1).expand_as(u)[edge], u[edge], v[edge]), delta[edge], accumulate=True)
            self._viz_record(iteration=season, best_profit=float(best_p.mean()))
        return best_S
