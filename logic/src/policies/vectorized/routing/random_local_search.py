"""Vectorised random local search: the simulator's RLS route improver on a batch.

Mirrors :class:`~logic.src.policies.route_improvement.random_local_search.RandomLocalSearchRouteImprover`
configured by ``RandomLocalSearchPostConfig``: ``iterations`` times, an operator is drawn
from ``params`` (normalised probabilities, NumPy ``default_rng(seed)`` exactly as the
simulator draws them) and one first-improvement sweep of that operator is applied, as
``SweepLocalSearch`` does (:func:`~logic.src.policies.vectorized.routing.shared.sweeps.sweep`);
time stops the search after ``time_limit`` seconds.

The starting solution is ``initial_solution`` when given, otherwise a random giant tour
over all bins cut into capacity-feasible routes.

Attributes:
    RandomLocalSearchPolicy: Batched RLS configured by ``RandomLocalSearchPostConfig``.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from logic.src.configs.policies.route_improvement import RandomLocalSearchPostConfig
from logic.src.policies.vectorized.routing.shared import routes as rs
from logic.src.policies.vectorized.routing.shared.base import VectorizedRoutingPolicy
from logic.src.policies.vectorized.routing.shared.sweeps import SweepState, sweep
from logic.src.policies.vectorized.routing.shared.vrpp import VRPPBatch


class RandomLocalSearchPolicy(VectorizedRoutingPolicy):
    """Batched counterpart of the simulator's ``random_local_search`` route improver.

    Takes exactly the ``RandomLocalSearchPostConfig`` fields (``iterations``, ``params``,
    ``time_limit``, ``seed``). Operator keys outside the simulator's op map are ignored,
    as the simulator does.
    """

    config_cls = RandomLocalSearchPostConfig

    def solve(self, inst: VRPPBatch, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the batched random local search.

        Args:
            inst: Instances.
            initial: Optional starting route sequences.

        Returns:
            torch.Tensor: Improved route sequences.
        """
        cfg = self.cfg
        if initial is None:
            tours = rs.random_giant_tours(inst.batch_size, inst.n_bins, self.generator, inst.device)
            S = rs.from_giant_tours(tours, inst)
        else:
            S = initial.clone()
        params = dict(cfg.params or {})
        active_ops = [op for op in params if op in rs.LS_OPERATORS]
        if not active_ops or inst.n_bins < 2:
            return S
        probs = np.array([params.get(op, 0.0) for op in active_ops], dtype=float)
        probs = probs / (probs.sum() + 1e-10)
        rng = np.random.default_rng(self.seed)
        start = time.perf_counter()
        state = SweepState.from_seqs(S)
        for it in range(int(cfg.iterations)):
            if cfg.time_limit > 0 and time.perf_counter() - start > cfg.time_limit:
                break
            op = str(rng.choice(active_ops, p=probs))
            state, _ = sweep(state, inst, op, self.generator)
            self._viz_record(iteration=it, op_name=op)
        return state.to_seqs(inst.n_bins)
