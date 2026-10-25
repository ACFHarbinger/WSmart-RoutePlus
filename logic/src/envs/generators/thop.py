"""Instance generator for the Thief Orienteering Problem.

The ThOP environment and task are in ``logic.src.envs.thop``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.envs.base.generator import Generator


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


MAX_TIMES: dict[int, float] = {20: 2.0, 50: 3.0, 100: 4.0}
