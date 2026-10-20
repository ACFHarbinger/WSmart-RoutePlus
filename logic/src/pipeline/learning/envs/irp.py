"""Inventory Routing Problem (IRP) task and environment on the routing base.

IRPEnv lives in ``irp_env``.

IRPGenerator lives in ``irp_generator``.

The module also holds ``IRPGenerator``, the problem's instance generator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.pipeline.learning.envs.base.problem import BaseProblem
from logic.src.pipeline.learning.envs.spec import PROBLEMS, ProblemSpec


class IRP(BaseProblem):
    """Inventory Routing Problem (IRP) task definition."""

    NAME = "irp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute IRP costs and outcomes."""
        routing_cost = dataset.get("routing_cost")
        if routing_cost is not None and isinstance(routing_cost, torch.Tensor):
            holding_cost = dataset.get("holding_cost", torch.zeros_like(routing_cost))
            stockout = dataset.get("stockout", torch.zeros_like(routing_cost))
            cost = routing_cost + holding_cost - stockout
        else:
            length = IRP.get_tour_length(dataset, pi, dist_matrix)
            cost = length
        return {
            "cost": cost,
            "reward": -cost,
            "length": cost,
            "valid": torch.ones_like(cost, dtype=torch.bool),
            "total": cost,
        }


IRP_SPEC = ProblemSpec(
    name="irp",
    objective=lambda cost, **kw: -float(cost),
    constraints=(),
    metrics=("cost", "reward", "length"),
    env_target="logic.src.pipeline.learning.envs.irp_env:IRPEnv",
    expand_pool=False,
)
PROBLEMS.register(IRP_SPEC)

__all__ = ["IRP", "IRP_SPEC"]
