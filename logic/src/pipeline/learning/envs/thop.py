"""Thief Orienteering Problem (ThOP) task and environment on the routing base.

The module also holds ``ThOPGenerator``, the problem's instance generator.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.pipeline.learning.envs.base.generator import Generator
from logic.src.pipeline.learning.envs.base.ops import OpsMixin
from logic.src.pipeline.learning.envs.base.problem import BaseProblem
from logic.src.pipeline.learning.envs.base.routing import RoutingEnv
from logic.src.pipeline.learning.envs.spec import PROBLEMS, ProblemSpec

# Default time-budget table (mirrors the OP max-length table)
MAX_TIMES: dict[int, float] = {20: 2.0, 50: 3.0, 100: 4.0}


class ThOPGenerator(Generator):
    """
    Generator for the Thief Orienteering Problem.

    Instance layout
    ---------------
    * **Locations** – ``num_loc`` customer cities plus one depot.
      ``locs[0]`` is always the depot; ``locs[1..num_loc]`` are the customer
      cities.
    * **Items** – ``num_items_per_city`` items at each customer city, giving
      ``num_items = num_loc * num_items_per_city`` items in total.  Items are
      assigned to cities sequentially: items ``0..K-1`` belong to city 1,
      items ``K..2K-1`` to city 2, …

    TensorDict fields
    -----------------
    ``locs``         : ``[*B, num_loc+1, 2]``  city coordinates (depot at 0)
    ``item_weights`` : ``[*B, num_items]``      item weights ∈ [min_weight, max_weight]
    ``item_profits`` : ``[*B, num_items]``      item profits ∈ [min_profit, max_profit]
    ``item_city``    : ``[*B, num_items]``      city index (1-based) for each item
    ``capacity``     : ``[*B]``                 knapsack weight capacity W
    ``max_time``     : ``[*B]``                 time budget T
    ``v_max``        : ``[*B]``                 maximum travel speed
    ``v_min``        : ``[*B]``                 minimum travel speed (at W = capacity)

    Attributes:
        num_loc: Number of customer locations.
        num_items_per_city: Number of items per city.
        min_loc: Lower bound for node coordinates.
        max_loc: Upper bound for node coordinates.
        loc_distribution: Spatial distribution strategy.
        min_weight: Minimum item weight.
        max_weight: Maximum item weight.
        min_profit: Minimum item profit.
        max_profit: Maximum item profit.
        capacity: Knapsack weight capacity.
        max_time: Time budget.
        v_max: Maximum travel speed.
        v_min: Minimum travel speed.
        depot_type: Depot placement strategy.
    """

    def __init__(
        self,
        num_loc: int = 20,
        num_items_per_city: int = 3,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        min_profit: float = 0.0,
        max_profit: float = 1.0,
        capacity: Optional[float] = None,
        max_time: Optional[float] = None,
        v_max: float = 1.0,
        v_min: float = 0.1,
        depot_type: str = "random",
        device: Union[str, torch.device] = "cpu",
        rng: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise ThOPGenerator.

        Args:
            num_loc: Number of customer cities (excluding depot).
            num_items_per_city: Items placed at each customer city.
                Total items = num_loc * num_items_per_city.
            min_loc: Lower bound for city coordinate values.
            max_loc: Upper bound for city coordinate values.
            loc_distribution: Spatial distribution for city placement
                ("uniform", "normal", "clustered", or callable).
            min_weight: Minimum item weight.
            max_weight: Maximum item weight.
            min_profit: Minimum item profit.
            max_profit: Maximum item profit.
            capacity: Knapsack capacity W. Defaults to ~25 % of total
                average item weight so the agent can carry roughly
                a quarter of all items.
            max_time: Time budget T. Defaults to the MAX_TIMES lookup
                table (same values as the OP max_length table).
            v_max: Maximum travel speed (when knapsack is empty).
            v_min: Minimum travel speed (when knapsack is full).
            depot_type: Depot placement strategy
                ("random", "center", "corner").
            device: Target torch device.
            rng: Optional numpy random generator.
            generator: Optional torch random generator.
            kwargs: Forwarded to Generator base class.
        """
        super().__init__(
            num_loc=num_loc,
            min_loc=min_loc,
            max_loc=max_loc,
            loc_distribution=loc_distribution,
            device=device,
            rng=rng,
            generator=generator,
            **kwargs,
        )
        self.num_items_per_city = num_items_per_city
        self.num_items = num_loc * num_items_per_city
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_profit = min_profit
        self.max_profit = max_profit
        self.v_max = v_max
        self.v_min = v_min
        self.depot_type = depot_type

        if capacity is None:
            avg_weight = (min_weight + max_weight) / 2.0
            capacity = self.num_items * avg_weight * 0.25
        self.capacity = float(capacity)

        if max_time is None:
            closest = min(MAX_TIMES.keys(), key=lambda x: abs(x - num_loc))
            max_time = MAX_TIMES.get(num_loc, MAX_TIMES[closest])
        self.max_time = float(max_time)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate a batch of ThOP instances.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with:
                locs: [*B, num_loc, 2]     — city coordinates (depot at 0)
                item_weights: [*B, num_items]  — item weights
                item_profits: [*B, num_items]  — item profits
                item_city: [*B, num_items]     — city index for each item
                capacity: [*B]               — knapsack capacity
                max_time: [*B]               — time budget
                v_max: [*B]                   — maximum travel speed
                v_min: [*B]                   — minimum travel speed
        """
        # City coordinates
        locs = self._generate_locations(batch_size)  # [*B, num_loc, 2]
        depot = self._generate_depot(batch_size)  # [*B, 2]
        locs_full = torch.cat([depot.unsqueeze(-2), locs], dim=-2)  # [*B, num_loc+1, 2]

        # Item weights and profits: uniform in their respective ranges
        item_weights = (
            torch.rand(*batch_size, self.num_items, device=self.device, generator=self.generator)
            * (self.max_weight - self.min_weight)
            + self.min_weight
        )
        item_profits = (
            torch.rand(*batch_size, self.num_items, device=self.device, generator=self.generator)
            * (self.max_profit - self.min_profit)
            + self.min_profit
        )

        # Item-to-city assignment: items are laid out consecutively per city.
        # city_k owns items [(k-1)*K, k*K) for K = num_items_per_city.
        # City indices are 1-based (depot = 0).
        item_city = torch.arange(1, self.num_loc + 1, dtype=torch.long, device=self.device).repeat_interleave(
            self.num_items_per_city
        )  # [num_items]
        item_city = item_city.unsqueeze(0).expand(*batch_size, -1).contiguous()  # [*B, num_items]

        # Scalar fields broadcast across the batch
        capacity = torch.full((*batch_size,), self.capacity, device=self.device, dtype=torch.float32)
        max_time = torch.full((*batch_size,), self.max_time, device=self.device, dtype=torch.float32)
        v_max = torch.full((*batch_size,), self.v_max, device=self.device, dtype=torch.float32)
        v_min = torch.full((*batch_size,), self.v_min, device=self.device, dtype=torch.float32)

        return TensorDict(
            {
                "locs": locs_full,
                "item_weights": item_weights,
                "item_profits": item_profits,
                "item_city": item_city,
                "capacity": capacity,
                "max_time": max_time,
                "v_max": v_max,
                "v_min": v_min,
            },
            batch_size=batch_size,
            device=self.device,
        )

    def _generate_depot(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate the depot location according to ``depot_type``.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with depot location (shape: [*B, 2]).
        """
        if self.depot_type == "center":
            center = (self.max_loc + self.min_loc) / 2.0
            return torch.full((*batch_size, 2), center, device=self.device, dtype=torch.float32)
        elif self.depot_type == "corner":
            return torch.full((*batch_size, 2), self.min_loc, device=self.device, dtype=torch.float32)
        else:  # "random"
            return self._uniform_locations(batch_size, 1).squeeze(-2)


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
    env_target="logic.src.pipeline.learning.envs.thop:ThOPEnv",
    expand_pool=True,
)
PROBLEMS.register(ThOP_SPEC)

__all__ = ["ThOPGenerator", "ThOPEnv", "ThOP", "ThOP_SPEC"]
