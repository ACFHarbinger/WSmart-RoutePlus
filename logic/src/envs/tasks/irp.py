"""Inventory Routing Problem (IRP) task and environment on ESPPRC base."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.espprc import ESPPRCEnv, ESPPRCGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class IRPGenerator(ESPPRCGenerator):
    """Generator for IRP instances."""

    def __init__(self, num_loc: int = 50, num_periods: int = 3, device: Union[str, torch.device] = "cpu", **kwargs: Any) -> None:
        super().__init__(num_loc=num_loc, device=device, **kwargs)
        self.num_periods = num_periods

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        td = super()._generate(batch_size)
        bs = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]
        td["demand"] = torch.rand((*bs, self.num_periods, self.num_loc), device=self.device)
        td["holding_cost"] = torch.rand((*bs, self.num_loc), device=self.device)
        td["capacity"] = torch.ones((*bs,), device=self.device)
        return td


class IRPEnv(ESPPRCEnv):
    """Inventory Routing Problem Environment on ESPPRC base."""

    NAME: str = "irp"
    name: str = "irp"
    node_dim: int = 4

    def __init__(
        self,
        generator: Optional[IRPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = IRPGenerator(**generator_params, device=device)
        super().__init__(generator=generator, generator_params=generator_params, device=device, **kwargs)


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
    env_target="logic.src.envs.tasks.irp:IRPEnv",
    expand_pool=False,
)
PROBLEMS.register(IRP_SPEC)

__all__ = ["IRPGenerator", "IRPEnv", "IRP", "IRP_SPEC"]
