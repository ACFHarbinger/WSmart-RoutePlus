"""Thief Orienteering Problem (ThOP) task and environment on the routing base.

ThOPEnv lives in ``thop_env``.

ThOPGenerator lives in ``thop_generator``.

The module also holds ``ThOPGenerator``, the problem's instance generator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from logic.src.pipeline.learning.envs.base.problem import BaseProblem
from logic.src.pipeline.learning.envs.spec import PROBLEMS, ProblemSpec

# Default time-budget table (mirrors the OP max-length table)


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
            return {
                "cost": z,
                "reward": z,
                "profit": z,
                "length": z,
                "time": z,
                "valid": torch.ones_like(z, dtype=torch.bool),
                "total": z,
            }

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
                dist_step = (
                    locs[torch.arange(B, device=device), city] - locs[torch.arange(B, device=device), cur_city]
                ).norm(dim=-1)
                v = v_max - (cur_weight / W) * (v_max - v_min)
                v = v.clamp(min=1e-5)
                cur_time = cur_time + torch.where(is_item, dist_step / v, torch.zeros_like(v))
                total_dist = total_dist + torch.where(is_item, dist_step, torch.zeros_like(dist_step))
                p = item_p.gather(1, item_idx.unsqueeze(1)).squeeze(1)
                w = item_w.gather(1, item_idx.unsqueeze(1)).squeeze(1)
                total_profit = total_profit + torch.where(is_item, p, torch.zeros_like(p))
                cur_weight = cur_weight + torch.where(is_item, w, torch.zeros_like(w))
                cur_city = torch.where(is_item, city, cur_city)

        return_dist = (locs[torch.arange(B, device=device), 0] - locs[torch.arange(B, device=device), cur_city]).norm(
            dim=-1
        )
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
    env_target="logic.src.pipeline.learning.envs.thop_env:ThOPEnv",
    expand_pool=True,
)
PROBLEMS.register(ThOP_SPEC)

__all__ = ["ThOP", "ThOP_SPEC"]
