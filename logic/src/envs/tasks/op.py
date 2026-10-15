"""Orienteering Problem (OP) task and environment on ESPPRC base."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.espprc import ESPPRCEnv, ESPPRCGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class OPGenerator(ESPPRCGenerator):
    """Generator for Orienteering Problem (OP) instances."""

    def __init__(
        self,
        num_loc: int = 50,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        max_length: float = 2.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(num_loc=num_loc, min_loc=min_loc, max_loc=max_loc, device=device, **kwargs)
        self.max_length = max_length

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        td = super()._generate(batch_size)
        bs = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]
        prize = torch.rand((*bs, self.num_loc), device=self.device)
        td["prize"] = prize
        td["max_length"] = torch.full((*bs,), self.max_length, device=self.device)
        return td


class OPEnv(ESPPRCEnv):
    """Orienteering Problem Environment on ESPPRC base."""

    NAME: str = "op"
    name: str = "op"
    node_dim: int = 3
    require_all_customers: bool = False

    def __init__(
        self,
        generator: Optional[OPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = OPGenerator(**generator_params, device=device)
        super().__init__(
            generator=generator,
            generator_params=generator_params,
            resources=(),
            waste_weight=1.0,
            cost_weight=0.0,
            device=device,
            **kwargs,
        )


class OP(BaseProblem):
    """Orienteering Problem (OP) task definition."""

    NAME = "op"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute OP prize, length, and outcomes."""
        if pi.size(-1) <= 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return {"cost": z, "reward": z, "prize": z, "length": z, "valid": torch.ones_like(z, dtype=torch.bool), "total": z}

        prize_all = dataset.get("prize", torch.zeros(pi.size(0), pi.size(1), device=pi.device))
        if prize_all.shape[-1] == pi.shape[-1] - 1:
            prize_all = torch.cat([torch.zeros_like(prize_all[:, :1]), prize_all], dim=-1)

        collected_prize = prize_all.gather(1, pi).sum(-1)
        length = OP.get_tour_length(dataset, pi, dist_matrix)
        neg_prize = -collected_prize

        max_len = dataset.get("max_length", 2.0)
        max_len_t = torch.as_tensor(max_len, dtype=torch.float32, device=pi.device).reshape(-1)
        if max_len_t.size(0) != pi.size(0):
            max_len_t = max_len_t.expand(pi.size(0))
        valid = length <= max_len_t + 1e-4

        return {
            "cost": neg_prize,
            "reward": collected_prize,
            "prize": collected_prize,
            "length": length,
            "valid": valid,
            "total": neg_prize,
        }


OP_SPEC = ProblemSpec(
    name="op",
    objective=lambda prize, **kw: float(prize),
    constraints=(),
    metrics=("cost", "reward", "prize", "length"),
    env_target="logic.src.envs.tasks.op:OPEnv",
    expand_pool=True,
)
PROBLEMS.register(OP_SPEC)

__all__ = ["OPGenerator", "OPEnv", "OP", "OP_SPEC"]
