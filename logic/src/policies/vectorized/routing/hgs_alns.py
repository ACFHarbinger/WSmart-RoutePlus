"""Vectorised HGS-ALNS policy: the simulator's HGS-ALNS (``HGSALNSConfig``) on a batch.

Mirrors :class:`~logic.src.policies.routing.meta_heuristics.hybrid_genetic_search_with_adaptive_large_neighborhood_search.hgs_alns.HGSALNSSolver`:

1. ``hgs.mu`` random giant tours, decoded by the profit Split (no education);
2. until ``hgs.n_iterations_no_improvement`` iterations pass without a better child
   (or ``time_limit`` seconds): binary tournament on biased fitness (diversity over the
   ``hgs.nb_granular`` closest individuals), ordered crossover, and with probability
   ``hgs.mutation_rate`` an ALNS run (``alns`` config) from the decoded child, the child's
   giant tour being rebuilt from the ALNS routes followed by the unvisited bins;
3. when the population exceeds ``2 * hgs.mu`` it is cut back to the ``hgs.mu`` fittest.

Attributes:
    VectorizedHGSALNS: Batched HGS-ALNS expert configured by ``HGSALNSConfig``.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from logic.src.configs.policies.hgs_alns import HGSALNSConfig
from logic.src.policies.vectorized.routing.core.crossover import vectorized_ordered_crossover
from logic.src.policies.vectorized.routing.shared import routes as rs
from logic.src.policies.vectorized.routing.shared.base import VectorizedRoutingPolicy
from logic.src.policies.vectorized.routing.shared.vrpp import VRPPBatch

from .adaptive_large_neighborhood_search import VectorizedALNSEngine
from .hybrid_genetic_search import HGSPopulation, decode, encode


class VectorizedHGSALNS(VectorizedRoutingPolicy):
    """Batched counterpart of the simulator's ``HGSALNSPolicy`` / ``HGSALNSSolver``.

    Takes exactly the ``HGSALNSConfig`` fields (nested ``hgs`` and ``alns`` configs);
    ``packing`` and ``route_improvement`` are simulator-only and ignored.
    """

    config_cls = HGSALNSConfig

    def solve(self, inst: VRPPBatch, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the batched HGS-ALNS.

        Args:
            inst: Instances.
            initial: Ignored (the search starts from its own population).

        Returns:
            torch.Tensor: Best route sequences.
        """
        cfg = self.cfg
        hgs = cfg.hgs
        B, N, M = inst.batch_size, inst.n_bins, inst.n_bins + 1
        dev = inst.device
        if N == 0:
            return rs.empty_routes(B, N, dev)
        mu = int(hgs.mu)
        pop = HGSPopulation(B, 2 * mu + 2, N, rs.width(N), M, dev)
        for _ in range(mu):
            t = rs.random_giant_tours(B, N, self.generator, dev)
            S = decode(t, inst, int(hgs.max_vehicles))
            pop.insert(t, S, rs.seq_profit(S, inst))
        _, best_S, best_p = pop.best()
        alns = VectorizedALNSEngine(cfg.alns, self.generator, self.expand_pool)
        no_imp = torch.zeros(B, dtype=torch.long, device=dev)
        start = time.perf_counter()
        it = 0
        r = torch.arange(B, device=dev)
        while True:
            running = no_imp < int(hgs.n_iterations_no_improvement)
            if not bool(running.any()):
                break
            if cfg.time_limit > 0 and time.perf_counter() - start >= cfg.time_limit:
                break
            it += 1
            fit = pop.biased_fitness(int(hgs.nb_elite), int(hgs.nb_granular))
            p1 = pop.tours[r, pop.tournament(fit, self.generator)]
            p2 = pop.tours[r, pop.tournament(fit, self.generator)]
            child = vectorized_ordered_crossover(p1, p2, generator=self.generator)
            S = decode(child, inst, int(hgs.max_vehicles))
            mutate = (torch.rand(B, generator=self.generator, device=dev) < hgs.mutation_rate) & (rs.n_visited(S) > 0)
            rows = (mutate & running).nonzero(as_tuple=True)[0]
            if rows.numel():
                S_m, _ = alns.solve(inst.subset(rows), S[rows])
                S[rows] = S_m
                child[rows] = encode(S_m, child[rows])
            p = rs.seq_profit(S, inst)
            ins = running.nonzero(as_tuple=True)[0]
            pop.insert(child[ins], S[ins], p[ins], ins)
            imp = running & (p > best_p)
            best_S = torch.where(imp.unsqueeze(1), S, best_S)
            best_p = torch.where(imp, p, best_p)
            no_imp = torch.where(imp, torch.zeros_like(no_imp), no_imp + running.long())
            over = pop.size() > 2 * mu
            if bool(over.any()):
                rows_o = over.nonzero(as_tuple=True)[0]
                fit = pop.biased_fitness(int(hgs.nb_elite), int(hgs.nb_granular))[rows_o]
                keep_rank = torch.argsort(torch.argsort(fit, dim=1, stable=True), dim=1)
                drop = pop.alive[rows_o] & (keep_rank >= mu)
                pop.alive[rows_o] = pop.alive[rows_o] & ~drop
                pop.profit[rows_o] = torch.where(
                    drop, torch.full_like(pop.profit[rows_o], float("-inf")), pop.profit[rows_o]
                )
            self._viz_record(iteration=it, best_profit=float(best_p.mean()))
        return best_S
