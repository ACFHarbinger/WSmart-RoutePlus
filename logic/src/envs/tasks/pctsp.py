"""Prize-Collecting Traveling Salesman Problem (PCTSP) task and environment on ESPPRC base."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.espprc import ESPPRCEnv, ESPPRCGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class PCTSPGenerator(ESPPRCGenerator):
    """Generator for PCTSP instances."""

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        min_prize: float = 0.0,
        max_prize: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(num_loc=num_loc, min_loc=min_loc, max_loc=max_loc, device=device, **kwargs)
        self.min_prize = min_prize
        self.max_prize = max_prize

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        td = super()._generate(batch_size)
        bs = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]
        deterministic_prize = torch.rand((*bs, self.num_loc), device=self.device)
        penalty = 3.0 * torch.rand((*bs, self.num_loc), device=self.device)
        td["deterministic_prize"] = deterministic_prize
        td["expected_prize"] = deterministic_prize
        td["real_prize"] = deterministic_prize
        td["penalty"] = penalty
        return td


class PCTSPEnv(ESPPRCEnv):
    """Prize-Collecting TSP Environment on ESPPRC base."""

    NAME: str = "pctsp"
    name: str = "pctsp"
    node_dim: int = 4
    require_all_customers: bool = False

    def __init__(
        self,
        generator: Optional[PCTSPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = PCTSPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=1.0,
            cost_weight=1.0,
            device=device,
            **kwargs,
        )


class PCTSP(BaseProblem):
    """Prize-Collecting TSP (PCTSP) task definition."""

    NAME = "pctsp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute PCTSP objective and outcomes."""
        if pi.size(-1) <= 1:
            pen = dataset["penalty"].sum(-1) if "penalty" in dataset else torch.zeros(pi.size(0), device=pi.device)
            return {"cost": pen, "reward": -pen, "saved_penalty": torch.zeros_like(pen), "length": torch.zeros_like(pen), "valid": torch.ones_like(pen, dtype=torch.bool), "total": pen}

        penalty = dataset.get("penalty", torch.zeros(pi.size(0), pi.size(1), device=pi.device))
        if penalty.shape[-1] == pi.max().item():
            penalty = torch.cat([torch.zeros_like(penalty[:, :1]), penalty], dim=-1)

        saved_penalty = penalty.gather(1, pi).sum(-1)
        length = PCTSP.get_tour_length(dataset, pi, dist_matrix)
        total_penalty = penalty.sum(-1)
        remaining_penalty = total_penalty - saved_penalty
        neg_reward = length + remaining_penalty

        return {
            "cost": neg_reward,
            "reward": -neg_reward,
            "saved_penalty": saved_penalty,
            "length": length,
            "valid": torch.ones_like(neg_reward, dtype=torch.bool),
            "total": neg_reward,
        }


PCTSP_SPEC = ProblemSpec(
    name="pctsp",
    objective=lambda neg_reward, **kw: -float(neg_reward),
    constraints=(),
    metrics=("cost", "reward", "saved_penalty", "length"),
    env_target="logic.src.envs.tasks.pctsp:PCTSPEnv",
    expand_pool=True,
)
PROBLEMS.register(PCTSP_SPEC)

__all__ = ["PCTSPGenerator", "PCTSPEnv", "PCTSP", "PCTSP_SPEC"]
