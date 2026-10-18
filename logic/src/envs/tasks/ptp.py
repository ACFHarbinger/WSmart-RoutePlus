"""Profitable Tour Problem (PTP) environment, generator, and task specification.

In PTP, the agent routes a vehicle from a depot to collect prizes / waste from an optional
subset of customer nodes, maximizing collected profit minus travel distance cost.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from logic.src.envs.base.routing import RoutingEnv
from logic.src.envs.generators.ptp import PTPGenerator
from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.common.released_specs import PTP_SPEC
from logic.src.envs.tasks.core.ptp import evaluate_profitable_tour
from logic.src.envs.tasks.spec import PROBLEMS


class PTPEnv(RoutingEnv):
    """Profitable Tour Problem Environment.

    Selects an optimal subset of profitable nodes to visit on the routing base
    without resource constraints.
    """

    NAME: str = "ptp"
    name: str = "ptp"
    node_dim: int = 4
    explicit_stop: bool = False
    require_all_customers: bool = False
    # PTP never offers an empty bin unless it is mandatory (the pre-WP-07 envs/routing/ptp.py).
    mask_empty_bins: bool = True

    def __init__(
        self,
        generator: Optional[PTPGenerator] = None,
        generator_params: Optional[dict] = None,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = PTPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=waste_weight,
            cost_weight=cost_weight,
            revenue_kg=revenue_kg,
            cost_km=cost_km,
            device=device,
            **kwargs,
        )


class PTP(BaseProblem):
    """Profitable Tour Problem (PTP) task definition."""

    NAME: str = "ptp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        cw_dict: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a completed PTP tour sequence."""
        return evaluate_profitable_tour(dataset, pi, dist_matrix=dist_matrix, cw_dict=cw_dict, **kwargs)


PROBLEMS.register(PTP_SPEC)

__all__ = ["PTPGenerator", "PTPEnv", "PTP", "PTP_SPEC"]
