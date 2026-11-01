"""Vectorised ALNS policy: the simulator's ALNS (``ALNSConfig``) on a batch.

Attributes:
    VectorizedALNS: Batched ALNS expert configured by ``ALNSConfig``.

Example:
    >>> policy = VectorizedALNS(env_name="cmptp", max_iterations=200, time_limit=5.0)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Optional

import torch

from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.policies.vectorized.routing.shared.base import VectorizedRoutingPolicy
from logic.src.policies.vectorized.routing.shared.vrpp import VRPPBatch

from .adaptive_large_neighborhood_search import VectorizedALNSEngine


class VectorizedALNS(VectorizedRoutingPolicy):
    """Batched counterpart of the simulator's ``ALNSPolicy`` / ``ALNSSolver``.

    Takes exactly the ``ALNSConfig`` fields. ``engine`` (``custom``/``package``/
    ``ortools``), ``packing`` and ``route_improvement`` are simulator-only and ignored:
    the custom engine is always used.
    """

    config_cls = ALNSConfig

    def solve(self, inst: VRPPBatch, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the batched ALNS.

        Args:
            inst: Instances.
            initial: Optional starting route sequences.

        Returns:
            torch.Tensor: Best route sequences.
        """
        engine = VectorizedALNSEngine(self.cfg, self.generator, self.expand_pool, self._viz_record)
        best, _ = engine.solve(inst, initial)
        return best
