"""Vectorised Iterated Local Search: the simulator's ILS (``ILSConfig``) on a batch.

Mirrors :class:`~logic.src.policies.routing.meta_heuristics.iterated_local_search.solver.ILSSolver`:

1. initial solution by ``build_greedy_routes``;
2. ``n_restarts`` times: a greedy descent of at most ``inner_iterations`` steps, each
   step applying one random low-level heuristic out of the first ``n_llh``
   (random/worst/cluster removal of ``n_removal`` bins + greedy/regret-2 repair) and
   keeping it only when it improves;
3. the configured acceptance criterion decides whether the descent's local optimum
   replaces the incumbent; the best solution is tracked;
4. perturbation: random removal of ``max(2, int(n * perturbation_strength))`` bins and
   greedy reinsertion, then the next restart;
5. stop after ``time_limit`` seconds.

Attributes:
    IteratedLocalSearchPolicy: Batched ILS expert configured by ``ILSConfig``.
"""

from __future__ import annotations

import time
from typing import Optional

import torch

from logic.src.configs.policies.ils import ILSConfig
from logic.src.policies.vectorized.routing.shared import routes as rs
from logic.src.policies.vectorized.routing.shared.acceptance import BatchedAcceptance
from logic.src.policies.vectorized.routing.shared.base import VectorizedRoutingPolicy
from logic.src.policies.vectorized.routing.shared.llh import LLH_NAMES, apply_llh
from logic.src.policies.vectorized.routing.shared.vrpp import VRPPBatch


class IteratedLocalSearchPolicy(VectorizedRoutingPolicy):
    """Batched counterpart of the simulator's ``ILSPolicy`` / ``ILSSolver``.

    Takes exactly the ``ILSConfig`` fields; ``engine``, ``packing`` and
    ``route_improvement`` are simulator-only and ignored.
    """

    config_cls = ILSConfig

    def solve(self, inst: VRPPBatch, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the batched ILS.

        Args:
            inst: Instances.
            initial: Optional starting route sequences.

        Returns:
            torch.Tensor: Best route sequences.
        """
        cfg = self.cfg
        pa = bool(cfg.profit_aware_operators)
        B = inst.batch_size
        dev = inst.device
        start = time.perf_counter()
        S = initial.clone() if initial is not None else rs.greedy_profit_construction(inst, self.generator)
        if inst.n_bins == 0:
            return S
        profit = rs.seq_profit(S, inst)
        best_S, best_profit = S.clone(), profit.clone()
        acc = BatchedAcceptance(cfg.acceptance_criterion, B)
        acc.setup(profit)
        n_llh = max(1, min(int(cfg.n_llh), len(LLH_NAMES)))

        def timed_out() -> bool:
            return cfg.time_limit > 0 and time.perf_counter() - start > cfg.time_limit

        for restart in range(int(cfg.n_restarts)):
            if timed_out():
                break
            inner_S, inner_p = S.clone(), profit.clone()
            active = torch.ones(B, dtype=torch.bool, device=dev)
            for _ in range(int(cfg.inner_iterations)):
                if timed_out() or not bool(active.any()):
                    break
                llh = torch.randint(0, n_llh, (B,), generator=self.generator, device=dev)
                cand = inner_S.clone()
                n = torch.full((B,), int(cfg.n_removal), device=dev, dtype=torch.long)
                for h in range(n_llh):
                    rows = ((llh == h) & active).nonzero(as_tuple=True)[0]
                    if rows.numel():
                        cand[rows] = apply_llh(
                            h, inner_S[rows], n[rows], inst.subset(rows), self.generator, pa, self.expand_pool
                        )
                cp = rs.seq_profit(cand, inst)
                improved = active & (cp > inner_p)
                inner_S = torch.where(improved.unsqueeze(1), cand, inner_S)
                inner_p = torch.where(improved, cp, inner_p)
                active = improved
            accepted = acc.accept(profit, inner_p, best_profit, restart, int(cfg.n_restarts))
            S = torch.where(accepted.unsqueeze(1), inner_S, S)
            profit = torch.where(accepted, inner_p, profit)
            better = accepted & (profit > best_profit)
            best_S = torch.where(better.unsqueeze(1), S, best_S)
            best_profit = torch.where(better, profit, best_profit)
            acc.step(profit, inner_p, accepted, best_profit, restart, int(cfg.n_restarts))
            # Perturbation (random removal + greedy repair) for the next restart.
            nv = rs.n_visited(S)
            pert = nv >= 4
            n_rm = torch.clamp((nv.float() * cfg.perturbation_strength).long(), min=2)
            P = apply_llh(
                0, S, torch.where(pert, n_rm, torch.zeros_like(n_rm)), inst, self.generator, pa, self.expand_pool
            )
            S = torch.where(pert.unsqueeze(1), P, S)
            profit = rs.seq_profit(S, inst)
            self._viz_record(iteration=restart, best_profit=float(best_profit.mean()))
        return best_S
