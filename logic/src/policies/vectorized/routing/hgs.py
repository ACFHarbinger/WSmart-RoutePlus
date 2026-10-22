"""Vectorised HGS policy: the simulator's HGS (``HGSConfig``) on a batch.

Attributes:
    VectorizedHGS: Batched HGS expert configured by ``HGSConfig``.
"""

from __future__ import annotations

from typing import Optional

import torch

from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig
from logic.src.policies.vectorized.routing.shared.base import VectorizedRoutingPolicy
from logic.src.policies.vectorized.routing.shared.vrpp import VRPPBatch

from .hybrid_genetic_search import VectorizedHGSEngine


class VectorizedHGS(VectorizedRoutingPolicy):
    """Batched counterpart of the simulator's ``HGSPolicy`` / ``HGSSolver``.

    Takes exactly the ``HGSConfig`` fields; ``engine`` (``custom``/``pyvrp``),
    ``packing``, ``route_improvement`` and ``acceptance_criterion`` (telemetry only in
    the simulator's HGS) are ignored.
    """

    config_cls = HGSConfig

    def solve(self, inst: VRPPBatch, initial: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the batched HGS (``initial`` is not used: HGS starts from its own population).

        Args:
            inst: Instances.
            initial: Ignored.

        Returns:
            torch.Tensor: Best route sequences.
        """
        best, _ = VectorizedHGSEngine(self.cfg, self.generator, self._viz_record).solve(inst)
        return best
