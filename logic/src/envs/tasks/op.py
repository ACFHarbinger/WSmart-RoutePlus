"""Orienteering Problem (OP) task and environment on ESPPRC base."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.ops import as_batch_nodes
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

    # OP dynamics of the pre-WP-07 envs/routing/op.py (review CR-12): a customer is
    # offered only if the tour can reach it and still return within ``max_length``,
    # and the reward is the collected prize.

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        if "visited" in tensordict.keys():
            return tensordict
        prize = tensordict["prize"]
        tensordict = super()._reset_instance(tensordict)
        locs = tensordict["locs"]
        if prize.shape[-1] == locs.shape[-2] - 1:
            prize = torch.cat([torch.zeros_like(prize[..., :1]), prize], dim=-1)
        tensordict["prize"] = prize
        max_length = tensordict["max_length"]
        if max_length.dim() > 1:
            max_length = max_length.squeeze(-1)
        # Budget left at each node once the return to the depot is reserved.
        dist_to_depot = (locs[..., 0:1, :] - locs).norm(p=2, dim=-1)
        tensordict["max_length"] = max_length.unsqueeze(-1) - dist_to_depot - 1e-6
        tensordict["current_total_prize"] = torch.zeros(*tensordict.batch_size, device=tensordict.device)
        return tensordict

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
        tensordict = super()._step_instance(tensordict)
        action = as_batch_nodes(tensordict["action"])
        prize_here = tensordict["prize"].gather(-1, action.unsqueeze(-1)).squeeze(-1)
        tensordict["current_total_prize"] = tensordict["current_total_prize"] + prize_here
        return tensordict

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
        current = tensordict["current_node"].squeeze(-1)
        locs = tensordict["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        dist_current_to_node = (locs - current_loc.unsqueeze(1)).norm(p=2, dim=-1)
        exceeds_budget = tensordict["tour_length"].unsqueeze(-1) + dist_current_to_node > tensordict["max_length"]
        mask = ~(tensordict["visited"] | exceeds_budget)
        mask[..., 0] = True
        return mask

    def _get_reward(self, tensordict: TensorDict, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        return tensordict["current_total_prize"]


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
