"""Thief Orienteering Problem (ThOP): environment, task and specification on the routing base.

The module holds ``ThOPEnv`` and ``ThOP``; the instance generator ``ThOPGenerator`` lives in
``logic.src.envs.generators.thop``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.base.problem import BaseProblem
from logic.src.envs.base.routing import RoutingEnv
from logic.src.envs.generators.thop import ThOPGenerator
from logic.src.envs.spec import PROBLEMS, ProblemSpec


class ThOPEnv(RoutingEnv):
    """Thief Orienteering Problem Environment on the routing base."""

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

    # Dynamics of the pre-WP-07 envs/routing/thop.py, kept verbatim (review CR-12, Q-8b).

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialise ThOP episode state from a generated instance.

        Args:
            td: TensorDict produced by the generator, containing ``locs``
                (customer coordinates) and ``depot`` tensors.

        Returns:
            TensorDict: Initialised state with ``current_node``, ``visited``,
                ``tour``, ``tour_length``, ``first_node``, ``reward``,
                ``terminated``, and ``truncated`` fields.
        """
        if "item_picked" in td.keys():
            return td

        device = td.device
        bs = td.batch_size
        if self.generator is None:
            raise ValueError("Generator is not initialized.")
        m = td["item_weights"].shape[-1]  # total number of items

        # Thief starts at depot (city 0), nothing picked, no time elapsed
        td["current_city"] = torch.zeros(*bs, dtype=torch.long, device=device)
        td["current_time"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["current_weight"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["item_picked"] = torch.zeros(*bs, m, dtype=torch.bool, device=device)

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["reward"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["terminated"] = torch.zeros(*bs, dtype=torch.bool, device=device)
        td["truncated"] = torch.zeros(*bs, dtype=torch.bool, device=device)

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        """Execute action and update state for ThOP.

        Args:
            td: Current state TensorDict containing ``action``, ``current_city``,
                ``current_time``, ``current_weight``, ``item_picked``, ``locs``,
                ``depot``, ``item_city``, ``item_weights``, ``capacity``,
                ``v_max``, and ``v_min`` fields.

        Returns:
            TensorDict: Updated state with ``current_time``, ``current_weight``,
                ``item_picked``, ``current_city``, ``tour``, ``tour_length``,
                ``reward``, ``terminated``, and ``truncated`` fields.
        """
        return OpsMixin._step(self, td)

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Execute one ThOP action.

        Action 0  → return to depot.
        Action k  → pick item k-1 (0-indexed); travel to its city first.

        Travel time uses the speed model v(w) = v_max - (w/W)*(v_max-v_min)
        evaluated at the weight *before* the item is added to the knapsack.
        Args:
            td: Current state TensorDict containing ``action``, ``current_city``,
                ``current_time``, ``current_weight``, ``item_picked``, ``locs``,
                ``depot``, ``item_city``, ``item_weights``, ``capacity``,
                ``v_max``, and ``v_min`` fields.

        Returns:
            TensorDict: Updated state with ``current_time``, ``current_weight``,
                ``item_picked``, ``current_city``, ``tour``, ``tour_length``,
                ``reward``, ``terminated``, and ``truncated`` fields.
        """
        action = td["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)

        # Whether this action returns to the depot
        at_depot_action = action == 0  # [B]

        # 0-indexed item to pick (clamped so depot action doesn't produce -1)
        item_idx = (action - 1).clamp(min=0)  # [B]

        # Resolve target city: item's city, or city 0 for depot return
        item_city = td["item_city"]  # [B, m]
        target_city_item = item_city.gather(-1, item_idx.unsqueeze(-1)).squeeze(-1)  # [B]
        target_city = torch.where(at_depot_action, torch.zeros_like(target_city_item), target_city_item)

        # Coordinates
        locs = td["locs"]  # [B, n+1, 2]
        current_city = td["current_city"]  # [B]

        # Current and target location: [B, 2]
        curr_loc = locs.gather(1, current_city[:, None, None].expand(-1, 1, 2)).squeeze(1)
        tgt_loc = locs.gather(1, target_city[:, None, None].expand(-1, 1, 2)).squeeze(1)

        dist = (tgt_loc - curr_loc).norm(p=2, dim=-1)  # [B]

        # Speed at current weight (before adding new item)
        capacity = td["capacity"]  # [B]
        v_max = td["v_max"]  # [B]
        v_min = td["v_min"]  # [B]
        current_weight = td["current_weight"]  # [B]
        speed = (v_max - (current_weight / capacity.clamp(min=1e-8)) * (v_max - v_min)).clamp(min=1e-8)

        time_cost = dist / speed  # [B]

        # Update elapsed time and position
        td["current_time"] = td["current_time"] + time_cost
        td["current_city"] = target_city

        # Update weight: only when picking an item (not depot return)
        item_weight_here = td["item_weights"].gather(-1, item_idx.unsqueeze(-1)).squeeze(-1)  # [B]
        weight_delta = torch.where(at_depot_action, torch.zeros_like(item_weight_here), item_weight_here)
        td["current_weight"] = current_weight + weight_delta

        # Mark the selected item as picked (only for non-depot actions)
        one_hot = torch.zeros_like(td["item_picked"]).scatter(-1, item_idx.unsqueeze(-1), True)
        td["item_picked"] = td["item_picked"] | (one_hot & ~at_depot_action.unsqueeze(-1))

        td["tour"] = torch.cat([td["tour"], action.unsqueeze(-1)], dim=-1)

        return td

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """Episode ends when the thief returns to the depot (city 0) after ≥1 step.

        Args:
            td: Current state TensorDict.

        Returns:
            torch.Tensor: Boolean tensor of shape ``(batch,)`` indicating terminated episodes.
        """
        i_raw = td.get("i", torch.zeros(td.batch_size, dtype=torch.long, device=td.device))
        step = i_raw.squeeze(-1) if i_raw.dim() > 1 else i_raw
        return (td["current_city"] == 0) & (step > 0)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute the valid-action mask of shape ``[B, num_items + 1]``.

        Item k (action k, 1-indexed) is valid when ALL of:
          (a) not yet picked,
          (b) adding its weight would not exceed knapsack capacity,
          (c) time to travel to its city + time to return to depot after
              picking it fits within the remaining time budget.

        The depot (action 0) is always valid once the episode has started
        (step > 0), and is forced valid if no items can be selected.

        Args:
            td: Current state TensorDict.

        Returns:
            torch.Tensor: Boolean mask of shape ``[batch, num_items + 1]`` where
                ``True`` indicates a valid action.
        """
        locs = td["locs"]  # [B, n+1, 2]
        item_city = td["item_city"]  # [B, m]
        item_weights = td["item_weights"]  # [B, m]
        item_picked = td["item_picked"]  # [B, m]
        capacity = td["capacity"]  # [B]
        max_time = td["max_time"]  # [B]
        v_max = td["v_max"]  # [B]
        v_min = td["v_min"]  # [B]
        current_city = td["current_city"]  # [B]
        current_weight = td["current_weight"]  # [B]
        current_time = td["current_time"]  # [B]

        # "i" is added by OpsMixin._reset *after* _get_action_mask is first called,
        # so we must default to 0 when it is absent (initial reset call).
        i_raw = td.get("i", torch.zeros(td.batch_size, dtype=torch.long, device=td.device))
        step = i_raw.squeeze(-1) if i_raw.dim() > 1 else i_raw

        m = item_weights.shape[-1]

        # --- Current location: [B, 2] ---
        curr_loc = locs.gather(1, current_city[:, None, None].expand(-1, 1, 2)).squeeze(1)

        # --- City coordinate for every item: [B, m, 2] ---
        item_city_locs = locs.gather(1, item_city.unsqueeze(-1).expand(-1, m, 2))

        # --- Depot coordinate: [B, 1, 2] ---
        depot_loc = locs[:, 0:1, :]

        # (a) Speed at current weight: [B] → broadcast to [B, 1]
        speed_now = (
            (v_max - (current_weight / capacity.clamp(min=1e-8)) * (v_max - v_min)).clamp(min=1e-8).unsqueeze(-1)
        )  # [B, 1]

        # (b) Distance from current position to each item city: [B, m]
        dist_to_items = (item_city_locs - curr_loc.unsqueeze(1)).norm(p=2, dim=-1)

        # (c) Time to reach each item city: [B, m]
        time_to_items = dist_to_items / speed_now

        # (d) Weight after picking each item: [B, m]
        weight_after = current_weight.unsqueeze(-1) + item_weights

        # (e) Speed after picking each item: [B, m]
        speed_after = (
            v_max.unsqueeze(-1)
            - (weight_after / capacity.unsqueeze(-1).clamp(min=1e-8)) * (v_max - v_min).unsqueeze(-1)
        ).clamp(min=1e-8)

        # (f) Distance from each item city back to the depot: [B, m]
        dist_items_to_depot = (item_city_locs - depot_loc).norm(p=2, dim=-1)

        # (g) Return travel time after picking each item: [B, m]
        time_return = dist_items_to_depot / speed_after

        # (h) Remaining time budget: [B, 1]
        remaining = (max_time - current_time).unsqueeze(-1)

        # --- Validity conditions for each item ---
        time_feasible = (time_to_items + time_return) <= remaining + 1e-6  # [B, m]
        under_cap = weight_after <= capacity.unsqueeze(-1) + 1e-6  # [B, m]
        item_mask = ~item_picked & under_cap & time_feasible  # [B, m]

        # --- Depot validity ---
        # Forced valid if step > 0, or if literally no item can be picked
        no_items_left = ~item_mask.any(dim=-1)  # [B]
        depot_valid = (step > 0) | no_items_left  # [B]

        # Full mask: [B, m+1]
        return torch.cat([depot_valid.unsqueeze(-1), item_mask], dim=-1)

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Total profit of all items collected during the episode.

        Args:
            td: Final state TensorDict containing ``item_profits`` and
                ``item_picked``.
            actions: Optional tour action sequence (unused).

        Returns:
            torch.Tensor: Total profit per batch element, shape ``(batch,)``.
        """
        return (td["item_profits"] * td["item_picked"].float()).sum(dim=-1)


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
    env_target="logic.src.envs.thop:ThOPEnv",
    expand_pool=True,
)
PROBLEMS.register(ThOP_SPEC)

__all__ = ["ThOPEnv", "ThOP", "ThOP_SPEC"]
