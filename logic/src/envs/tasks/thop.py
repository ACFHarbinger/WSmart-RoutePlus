"""Thief Orienteering Problem (ThOP) task and environment on ESPPRC base."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from tensordict import TensorDict

from logic.src.envs.tasks.base import BaseProblem
from logic.src.envs.tasks.espprc import ESPPRCEnv, ESPPRCGenerator
from logic.src.envs.tasks.spec import PROBLEMS, ProblemSpec


class ThOPGenerator(ESPPRCGenerator):
    """Generator for ThOP instances."""

    def __init__(
        self,
        num_loc: int = 50,
        num_items: int = 100,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(num_loc=num_loc, device=device, **kwargs)
        self.num_items = num_items

    def _generate(self, batch_size: Union[int, Sequence[int]]) -> TensorDict:
        td = super()._generate(batch_size)
        bs = list(batch_size) if isinstance(batch_size, (list, tuple)) else [batch_size]
        td["item_weights"] = torch.rand((*bs, self.num_items), device=self.device)
        td["item_profits"] = torch.rand((*bs, self.num_items), device=self.device)
        td["item_city"] = torch.randint(1, self.num_loc + 1, size=(*bs, self.num_items), device=self.device)
        td["capacity"] = torch.ones((*bs,), device=self.device) * 10.0
        td["max_time"] = torch.ones((*bs,), device=self.device) * 20.0
        td["v_max"] = torch.ones((*bs,), device=self.device)
        td["v_min"] = torch.ones((*bs,), device=self.device) * 0.1
        return td


class ThOPEnv(ESPPRCEnv):
    """Thief Orienteering Problem Environment on ESPPRC base."""

    NAME: str = "thop"
    name: str = "thop"
    node_dim: int = 4

    def __init__(
        self,
        generator: Optional[ThOPGenerator] = None,
        generator_params: Optional[dict] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = ThOPGenerator(**generator_params, device=device)
        super().__init__(generator=generator, generator_params=generator_params, device=device, **kwargs)


class ThOP(BaseProblem):
    """Thief Orienteering Problem (ThOP) task definition."""

    NAME = "thop"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate a ThOP solution."""
        if pi.size(-1) <= 1:
            z = torch.zeros(pi.size(0), device=pi.device)
            return {"cost": z, "reward": z, "profit": z, "length": z, "time": z, "valid": torch.ones_like(z, dtype=torch.bool), "total": z}

        B, T = pi.shape
        device = pi.device
        locs = dataset["locs"]
        item_w = dataset["item_weights"]
        item_p = dataset["item_profits"]
        item_c = dataset["item_city"]
        W = dataset["capacity"]
        T_max = dataset["max_time"]
        v_max = dataset["v_max"]
        v_min = dataset["v_min"]

        total_profit = torch.zeros(B, device=device)
        cur_weight = torch.zeros(B, device=device)
        cur_time = torch.zeros(B, device=device)
        cur_city = torch.zeros(B, dtype=torch.long, device=device)
        total_dist = torch.zeros(B, device=device)

        for step in range(T):
            action = pi[:, step]
            is_depot = action == 0
            is_item = ~is_depot
            if is_item.any():
                item_idx = (action - 1).clamp(min=0)
                city = item_c.gather(1, item_idx.unsqueeze(1)).squeeze(1)
                dist_step = (locs[torch.arange(B, device=device), city] - locs[torch.arange(B, device=device), cur_city]).norm(dim=-1)
                v = v_max - (cur_weight / W) * (v_max - v_min)
                v = v.clamp(min=1e-5)
                cur_time = cur_time + torch.where(is_item, dist_step / v, torch.zeros_like(v))
                total_dist = total_dist + torch.where(is_item, dist_step, torch.zeros_like(dist_step))
                p = item_p.gather(1, item_idx.unsqueeze(1)).squeeze(1)
                w = item_w.gather(1, item_idx.unsqueeze(1)).squeeze(1)
                total_profit = total_profit + torch.where(is_item, p, torch.zeros_like(p))
                cur_weight = cur_weight + torch.where(is_item, w, torch.zeros_like(w))
                cur_city = torch.where(is_item, city, cur_city)

        return_dist = (locs[torch.arange(B, device=device), 0] - locs[torch.arange(B, device=device), cur_city]).norm(dim=-1)
        v = (v_max - (cur_weight / W) * (v_max - v_min)).clamp(min=1e-5)
        cur_time = cur_time + return_dist / v
        total_dist = total_dist + return_dist

        valid = (cur_time <= T_max + 1e-4) & (cur_weight <= W + 1e-4)
        neg_profit = -total_profit
        return {
            "cost": neg_profit,
            "reward": total_profit,
            "profit": total_profit,
            "length": total_dist,
            "time": cur_time,
            "valid": valid,
            "total": neg_profit,
        }


ThOP_SPEC = ProblemSpec(
    name="thop",
    objective=lambda profit, **kw: float(profit),
    constraints=(),
    metrics=("cost", "reward", "profit", "length", "time"),
    env_target="logic.src.envs.tasks.thop:ThOPEnv",
    expand_pool=True,
)
PROBLEMS.register(ThOP_SPEC)

__all__ = ["ThOPGenerator", "ThOPEnv", "ThOP", "ThOP_SPEC"]
