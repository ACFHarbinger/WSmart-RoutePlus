"""Pickup and Delivery Problem (PDP) task and environment on ESPPRC base."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.espprc import ESPPRCEnv, ESPPRCGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class PDPGenerator(ESPPRCGenerator):
    """Generator for Pickup and Delivery Problem (PDP) instances."""

    def __init__(self, num_loc: int = 50, device: Union[str, torch.device] = "cpu", **kwargs: Any) -> None:
        super().__init__(num_loc=num_loc, device=device, **kwargs)

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        td = super()._generate(batch_size)
        bs = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]
        capacity = 1.0
        td["capacity"] = torch.full((*bs,), capacity, device=self.device)
        return td


class PDPEnv(ESPPRCEnv):
    """Pickup and Delivery Problem Environment on ESPPRC base."""

    NAME: str = "pdp"
    name: str = "pdp"
    node_dim: int = 3
    require_all_customers: bool = True

    def __init__(
        self,
        generator: Optional[PDPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = PDPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=0.0,
            cost_weight=1.0,
            device=device,
            **kwargs,
        )


class PDP(BaseProblem):
    """Pickup and Delivery Problem (PDP) task definition."""

    NAME = "pdp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute PDP tour length and outcomes."""
        if pi.size(-1) <= 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return {"cost": z, "reward": z, "length": z, "valid": torch.ones_like(z, dtype=torch.bool), "total": z}

        length = PDP.get_tour_length(dataset, pi, dist_matrix)
        return {
            "cost": length,
            "reward": -length,
            "length": length,
            "valid": torch.ones_like(length, dtype=torch.bool),
            "total": length,
        }


PDP_SPEC = ProblemSpec(
    name="pdp",
    objective=lambda distance_km, **kw: -float(distance_km),
    constraints=(),
    metrics=("cost", "reward", "length"),
    env_target="logic.src.envs.tasks.pdp:PDPEnv",
    expand_pool=False,
)
PROBLEMS.register(PDP_SPEC)

__all__ = ["PDPGenerator", "PDPEnv", "PDP", "PDP_SPEC"]
