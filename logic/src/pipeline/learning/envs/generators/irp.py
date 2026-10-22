"""Instance generator for the Inventory Routing Problem.

The IRP environment and task are in ``logic.src.pipeline.learning.envs.irp``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from tensordict import TensorDict

from logic.src.pipeline.learning.envs.base.generator import Generator


class IRPGenerator(Generator):
    """
    Generator for deterministic Inventory Routing Problem (IRP) instances.

    Generates multi-period instances where a depot-based vehicle must make
    replenishment deliveries to customers over a planning horizon T to minimise
    the sum of routing cost and inventory holding cost:

        min_{t in T} ( sum_{(i,j) in A} c_ij x_ijt + sum_{i in V} h_i I_it )

    subject to inventory flow conservation:

        I_it = I_{i,t-1} + q_it - d_it,  0 <= I_it <= C_i  for all t

    Instance layout
    ---------------
    - locs            : [*B, num_loc, 2]       – customer coordinates
    - depot           : [*B, 2]                – depot coordinate
    - demands         : [*B, num_loc]          – deterministic daily demand d_i
    - holding_costs   : [*B, num_loc]          – unit holding cost h_i per period
    - initial_inventory: [*B, num_loc]         – starting inventory I_{i,0}
    - inventory_capacity: [*B, num_loc]        – node capacity C_i
    - vehicle_capacity  : [*B]                 – vehicle capacity Q

    The planning horizon T = num_periods is stored as an integer attribute and is
    shared across the batch (all instances in one batch have the same T).

    Attributes:
        num_periods: Planning horizon T.
        vehicle_capacity: Vehicle capacity Q (normalised).
        min_demand: Minimum per-period demand rate.
        max_demand: Maximum per-period demand rate.
        demand_distribution: Demand distribution strategy.
        min_holding_cost: Minimum unit holding cost.
        max_holding_cost: Maximum unit holding cost.
        min_init_inventory: Minimum initial inventory fraction.
        max_init_inventory: Maximum initial inventory fraction.
        node_inventory_capacity: Node inventory capacity C_i.
        depot_type: Depot placement strategy.
    """

    def __init__(
        self,
        num_loc: int = 20,
        num_periods: int = 5,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        loc_distribution: Union[str, Callable] = "uniform",
        vehicle_capacity: float = 1.0,
        min_demand: float = 0.05,
        max_demand: float = 0.20,
        demand_distribution: str = "uniform",
        min_holding_cost: float = 0.1,
        max_holding_cost: float = 1.0,
        min_init_inventory: float = 0.0,
        max_init_inventory: float = 0.5,
        node_inventory_capacity: float = 1.0,
        depot_type: str = "corner",
        device: Union[str, torch.device] = "cpu",
        rng: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise IRPGenerator.



        Args:
            num_loc: Number of customer locations (excluding depot).
            num_periods: Planning horizon T (number of delivery periods).
            min_loc: Lower bound for node coordinates.
            max_loc: Upper bound for node coordinates.
            loc_distribution: Spatial distribution ("uniform", "normal", "clustered").
            vehicle_capacity: Vehicle capacity Q (normalised).
            min_demand: Minimum per-period demand rate (fraction of node capacity).
            max_demand: Maximum per-period demand rate (fraction of node capacity).
            demand_distribution: Demand distribution ("uniform").
            min_holding_cost: Minimum unit holding cost h_i per period.
            max_holding_cost: Maximum unit holding cost h_i per period.
            min_init_inventory: Minimum initial inventory as fraction of C_i.
            max_init_inventory: Maximum initial inventory as fraction of C_i.
            node_inventory_capacity: Node inventory capacity C_i (homogeneous).
            depot_type: Depot placement strategy ("center", "corner", "random").
            device: Target device for generated tensors.
            rng: Optional numpy RNG.
            generator: Optional torch RNG.
            kwargs: Forwarded to Generator base.
        """
        super().__init__(num_loc, min_loc, max_loc, loc_distribution, device, rng, generator, **kwargs)

        self.num_periods = num_periods
        self.vehicle_capacity = vehicle_capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.demand_distribution = demand_distribution
        self.min_holding_cost = min_holding_cost
        self.max_holding_cost = max_holding_cost
        self.min_init_inventory = min_init_inventory
        self.max_init_inventory = max_init_inventory
        self.node_inventory_capacity = node_inventory_capacity
        self.depot_type = depot_type

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _generate(self, batch_size: tuple[int, ...]) -> TensorDict:
        """Generate a batch of IRP instances.

        Args:
            batch_size: Batch size.

        Returns:
            TensorDict with:
                locs: [*B, num_loc, 2]           — customer coordinates
                depot: [*B, 2]                    — depot coordinate
                demands: [*B, num_loc]            — per-period demand rates d_i
                holding_costs: [*B, num_loc]      — unit holding costs h_i
                initial_inventory: [*B, num_loc]  — initial inventories I_{i,0}
                inventory_capacity: [*B, num_loc] — node capacities C_i
                vehicle_capacity: [*B]           — vehicle capacity Q
                num_periods: int                  — planning horizon T
        """
        locs = self._generate_locations(batch_size)
        depot = self._generate_depot(batch_size)
        demands = self._generate_demands(batch_size)
        holding_costs = self._generate_holding_costs(batch_size)
        initial_inventory = self._generate_initial_inventory(batch_size)
        inventory_capacity = torch.full(
            (*batch_size, self.num_loc),
            self.node_inventory_capacity,
            device=self.device,
            dtype=torch.float32,
        )
        vehicle_capacity = torch.full(
            (*batch_size,),
            self.vehicle_capacity,
            device=self.device,
            dtype=torch.float32,
        )

        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "demands": demands,
                "holding_costs": holding_costs,
                "initial_inventory": initial_inventory,
                "inventory_capacity": inventory_capacity,
                "vehicle_capacity": vehicle_capacity,
            },
            batch_size=batch_size,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Field generators
    # ------------------------------------------------------------------

    def _generate_depot(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate depot location.

        Args:
            batch_size: Batch size.

        Returns:
            Depot location.
        """
        if self.depot_type == "center":
            center = (self.max_loc + self.min_loc) / 2.0
            return torch.full((*batch_size, 2), center, device=self.device, dtype=torch.float32)
        elif self.depot_type == "corner":
            return torch.full((*batch_size, 2), self.min_loc, device=self.device, dtype=torch.float32)
        elif self.depot_type == "random":
            return self._uniform_locations(batch_size, 1).squeeze(-2)
        else:
            raise ValueError(f"Unknown depot_type '{self.depot_type}'. Choose: center, corner, random.")

    def _generate_demands(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate per-node daily demand rates d_i (constant across periods).

        Args:
            batch_size: Batch size.

        Returns:
            Per-node daily demand rates.
        """
        if self.demand_distribution == "uniform":
            d = (
                torch.rand(*batch_size, self.num_loc, device=self.device, generator=self.generator)
                * (self.max_demand - self.min_demand)
                + self.min_demand
            )
        else:
            raise ValueError(f"Unknown demand_distribution '{self.demand_distribution}'.")

        # Clamp so demand never exceeds node capacity (would cause perpetual stockout)
        return d.clamp(max=self.node_inventory_capacity)

    def _generate_holding_costs(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate per-node unit holding costs h_i.

        Args:
            batch_size: Batch size.

        Returns:
            Holding costs for each node.
        """
        return (
            torch.rand(*batch_size, self.num_loc, device=self.device, generator=self.generator)
            * (self.max_holding_cost - self.min_holding_cost)
            + self.min_holding_cost
        )

    def _generate_initial_inventory(self, batch_size: tuple[int, ...]) -> torch.Tensor:
        """Generate initial inventory I_{i,0} as a fraction of C_i.

        Args:
            batch_size: Batch size.

        Returns:
            Initial inventory for each node.
        """
        frac = (
            torch.rand(*batch_size, self.num_loc, device=self.device, generator=self.generator)
            * (self.max_init_inventory - self.min_init_inventory)
            + self.min_init_inventory
        )
        return (frac * self.node_inventory_capacity).clamp(0.0, self.node_inventory_capacity)
