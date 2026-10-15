"""Asymmetric Traveling Salesman Problem (ATSP) task and environment."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.espprc import ESPPRCEnv, ESPPRCGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class ATSPGenerator(ESPPRCGenerator):
    """Generator for ATSP instances with asymmetric cost matrices."""

    def __init__(self, num_loc: int = 50, device: Union[str, torch.device] = "cpu", **kwargs: Any) -> None:
        super().__init__(num_loc=num_loc, device=device, **kwargs)

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        td = super()._generate(batch_size)
        bs = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]
        cost_matrix = torch.rand((*bs, self.num_loc, self.num_loc), device=self.device)
        diag_idx = torch.arange(self.num_loc, device=self.device)
        cost_matrix[..., diag_idx, diag_idx] = 0.0
        td["cost_matrix"] = cost_matrix
        return td


class ATSPEnv(ESPPRCEnv):
    """Asymmetric Traveling Salesman Problem Environment."""

    NAME: str = "atsp"
    name: str = "atsp"
    node_dim: int = 2
    require_all_customers: bool = True

    def __init__(
        self,
        generator: Optional[ATSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = ATSPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=0.0,
            cost_weight=1.0,
            device=device,
            **kwargs,
        )


class ATSP(BaseProblem):
    """Asymmetric Traveling Salesman Problem (ATSP) task definition."""

    NAME = "atsp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute ATSP tour cost and outcomes."""
        cost_matrix = dist_matrix if dist_matrix is not None else dataset.get("cost_matrix")
        if cost_matrix is None:
            raise ValueError("ATSP evaluation requires 'cost_matrix'")
        bs = cost_matrix.shape[0]
        if pi.dim() == 1:
            pi = pi.unsqueeze(0).expand(bs, -1)

        nodes_src = pi
        nodes_tgt = torch.roll(pi, -1, dims=-1)
        batch_idx = torch.arange(bs, device=cost_matrix.device).unsqueeze(1)
        cost = cost_matrix[batch_idx, nodes_src, nodes_tgt].sum(-1)

        return {
            "cost": cost,
            "reward": -cost,
            "length": cost,
            "valid": torch.ones_like(cost, dtype=torch.bool),
            "total": cost,
        }


ATSP_SPEC = ProblemSpec(
    name="atsp",
    objective=lambda distance_km, **kw: -float(distance_km),
    constraints=(),
    metrics=("cost", "reward", "length"),
    env_target="logic.src.envs.tasks.atsp:ATSPEnv",
    expand_pool=False,
)
PROBLEMS.register(ATSP_SPEC)

__all__ = ["ATSPGenerator", "ATSPEnv", "ATSP", "ATSP_SPEC"]
