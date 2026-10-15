"""Traveling Salesman Problem (TSP) task, environment, and generator."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.espprc import ESPPRCEnv, ESPPRCGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class TSPGenerator(ESPPRCGenerator):
    """Generator for TSP instances."""
    pass


class TSPEnv(ESPPRCEnv):
    """Traveling Salesman Problem Environment on the ESPPRC base."""

    NAME: str = "tsp"
    name: str = "tsp"
    node_dim: int = 2
    require_all_customers: bool = True

    def __init__(
        self,
        generator: Optional[TSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = TSPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=0.0,
            cost_weight=1.0,
            device=device,
            **kwargs,
        )


class TSP(BaseProblem):
    """Traveling Salesman Problem (TSP) task definition."""

    NAME = "tsp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute TSP tour cost and outcomes."""
        if pi.size(-1) <= 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return {"cost": z, "reward": z, "length": z, "valid": torch.ones_like(z, dtype=torch.bool), "total": z}

        if dist_matrix is not None and isinstance(dist_matrix, torch.Tensor):
            length = TSP._cost_from_matrix(pi, dist_matrix)
        elif "cost_matrix" in dataset:
            length = TSP._cost_from_matrix(pi, dataset["cost_matrix"])
        else:
            length = TSP._cost_from_locs(pi, dataset["locs"])

        return {
            "cost": length,
            "reward": -length,
            "length": length,
            "valid": torch.ones_like(length, dtype=torch.bool),
            "total": length,
        }

    @staticmethod
    def _cost_from_locs(pi: torch.Tensor, locs: torch.Tensor) -> torch.Tensor:
        visited = locs.gather(1, pi.unsqueeze(-1).expand(*pi.shape, locs.shape[-1]))
        closed = torch.cat([visited, visited[:, :1]], dim=1)
        return (closed[:, 1:] - closed[:, :-1]).norm(dim=-1).sum(dim=-1)

    @staticmethod
    def _cost_from_matrix(pi: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        B, N = pi.shape
        if mat.dim() == 2:
            mat = mat.unsqueeze(0).expand(B, -1, -1)
        pi_closed = torch.cat([pi, pi[:, :1]], dim=1)
        src = pi_closed[:, :-1]
        dst = pi_closed[:, 1:]
        batch_idx = torch.arange(B, device=pi.device).unsqueeze(1).expand(B, N)
        return mat[batch_idx, src, dst].sum(dim=-1)


TSP_SPEC = ProblemSpec(
    name="tsp",
    objective=lambda distance_km, **kw: -float(distance_km),
    constraints=(),
    metrics=("cost", "reward", "length"),
    env_target="logic.src.envs.tasks.tsp:TSPEnv",
    expand_pool=False,
)
PROBLEMS.register(TSP_SPEC)

__all__ = ["TSPGenerator", "TSPEnv", "TSP", "TSP_SPEC"]
