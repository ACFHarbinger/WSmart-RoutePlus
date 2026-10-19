"""Inventory Routing Problem (IRP) task and environment on the routing base.

The module also holds ``IRPGenerator``, the problem's instance generator.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.pipeline.rl.envs.base.generator import Generator
from logic.src.pipeline.rl.envs.base.ops import OpsMixin
from logic.src.pipeline.rl.envs.base.problem import BaseProblem
from logic.src.pipeline.rl.envs.base.routing import RoutingEnv
from logic.src.pipeline.rl.envs.spec import PROBLEMS, ProblemSpec


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


class IRPEnv(RoutingEnv):
    """Inventory Routing Problem Environment on the routing base."""

    NAME: str = "irp"
    name: str = "irp"
    node_dim: int = 2

    def __init__(
        self,
        generator: Optional[IRPGenerator] = None,
        generator_params: Optional[dict] = None,
        stockout_penalty: float = 10.0,
        holding_cost_weight: float = 1.0,
        routing_cost_weight: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> None:
        generator_params = generator_params or kwargs
        if generator is None:
            generator = IRPGenerator(**generator_params, device=device)
        super().__init__(generator=generator, generator_params=generator_params, device=device, **kwargs)
        self.stockout_penalty = stockout_penalty
        self.holding_cost_weight = holding_cost_weight
        self.routing_cost_weight = routing_cost_weight
        self.num_periods: int = generator.num_periods

    # Dynamics of the pre-WP-07 envs/routing/irp.py, kept verbatim (review CR-12, Q-8b).

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """
        Initialise per-episode state from a raw problem instance.

        Prepends the depot to ``locs`` when necessary and constructs all
        mutable state tensors required by the IRP.

        State fields added
        ------------------
        current_node       : LongTensor [*B]          – current vehicle position (starts at 0)
        visited            : BoolTensor [*B, N+1]     – per-period visit mask (depot always True)
        tour               : LongTensor [*B, 0]       – empty action sequence log
        tour_length        : Tensor     [*B]          – accumulated routing distance
        remaining_capacity : Tensor     [*B]          – remaining vehicle capacity
        current_inventory  : Tensor     [*B, N]       – current inventory I_it (= initial at t=0)
        delivered          : Tensor     [*B, N]       – delivered quantities this period
        current_period     : LongTensor [*B]          – period index in {0, …, T}
        total_holding_cost : Tensor     [*B]          – sum of h_i I_it over completed periods
        total_stockout     : Tensor     [*B]          – sum of max(0, -I_it) over completed periods

        Args:
            td: Input TensorDict containing graph structure and node properties.

        Returns:
            TensorDict: Initialized IRP state with per-episode mutable fields.
        """
        if "current_node" in td.keys():
            return td

        if self.generator is None:
            raise ValueError("Generator is not initialized.")

        device = td.device
        bs = td.batch_size

        # ----------------------------------------------------------
        # Ensure depot is prepended to locs
        # ----------------------------------------------------------
        locs = td["locs"]
        gen_n: Optional[int] = getattr(self.generator, "num_loc", None)
        needs_prepend = False

        if "depot" in td.keys():
            if gen_n is not None and locs.shape[-2] == gen_n + 1:
                needs_prepend = False
            elif (
                gen_n is not None
                and locs.shape[-2] == gen_n
                or not torch.allclose(locs[..., 0, :], td["depot"], atol=1e-4)
            ):
                needs_prepend = True

        if needs_prepend:
            td["locs"] = torch.cat([td["depot"].unsqueeze(-2), locs], dim=-2)

        num_nodes = td["locs"].shape[-2]  # num_loc + 1  (index 0 = depot)

        # ----------------------------------------------------------
        # Core routing state
        # ----------------------------------------------------------
        td["current_node"] = torch.zeros(*bs, dtype=torch.long, device=device)
        visited = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        visited[..., 0] = True  # depot always considered visited
        td["visited"] = visited

        td["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        td["tour_length"] = torch.zeros(*bs, dtype=torch.float32, device=device)

        # ----------------------------------------------------------
        # Vehicle capacity
        # ----------------------------------------------------------
        td["remaining_capacity"] = td["vehicle_capacity"].clone()

        # ----------------------------------------------------------
        # Inventory state (customer nodes only; index 0 = depot excluded)
        # ----------------------------------------------------------
        td["current_inventory"] = td["initial_inventory"].clone()
        td["delivered"] = torch.zeros(*bs, num_nodes - 1, dtype=torch.float32, device=device)

        # ----------------------------------------------------------
        # Period / cost accumulators
        # ----------------------------------------------------------
        td["current_period"] = torch.zeros(*bs, dtype=torch.long, device=device)
        td["total_holding_cost"] = torch.zeros(*bs, dtype=torch.float32, device=device)
        td["total_stockout"] = torch.zeros(*bs, dtype=torch.float32, device=device)

        return td

    def _step(self, td: TensorDict) -> TensorDict:
        """Execute one step in the environment.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            TensorDict: Updated TensorDict with new state and reward information.
        """
        return OpsMixin._step(self, td)

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """
        Execute one routing action and handle IRP-specific state transitions.

        Processing order
        ----------------
        1. Normalise *action* / *current_node* shapes.
        2. Detect whether this step is a genuine period-end depot return
           (vehicle returning from a customer to the depot).
        3. Call the base _step_instance to update distance, visited mask,
           current_node, and tour log.
        4. If visiting a customer: compute and apply the maximum feasible
           delivery, updating delivered[] and remaining_capacity.
        5. If period ends (depot return from customer): update inventories,
           accumulate holding cost / stockout, advance current_period, and
           reset per-period buffers (delivered, remaining_capacity, visited).

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            TensorDict: Updated TensorDict with new state and reward information.
        """
        action = td["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        current_node = td["current_node"]
        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)
        if current_node.dim() == 0:
            current_node = current_node.unsqueeze(0)

        # A genuine period-end: returning to depot from a customer node,
        # within an active (not yet completed) period.
        at_depot = action == 0  # [*B]
        from_customer = current_node != 0  # [*B]
        within_horizon = td["current_period"] < self.num_periods  # [*B]
        period_end = at_depot & from_customer & within_horizon  # [*B]

        # Base routing update: distance, visited, current_node, tour
        td = super()._step_instance(td)

        # Override current_node shape (base sets [*B, 1]; keep [*B])
        cn = td["current_node"]
        if cn.dim() > len(td.batch_size):
            td["current_node"] = cn.squeeze(-1)

        # ----------------------------------------------------------
        # Customer delivery
        # ----------------------------------------------------------
        is_customer = ~at_depot  # [*B]
        if is_customer.any():
            # Customer index into the 0-based customer arrays (action 1..N → 0..N-1)
            cust_idx = (action - 1).clamp(min=0)  # [*B]

            current_inv = td["current_inventory"]  # [*B, N]
            inv_cap = td["inventory_capacity"]  # [*B, N]
            remaining_cap = td["remaining_capacity"]  # [*B]

            # Space available at each node: C_i - I_i, clamped non-negative
            inv_space = (inv_cap - current_inv).clamp(min=0.0)  # [*B, N]

            # Gather the space at the visited node
            gather_idx = cust_idx.view(*td.batch_size, 1)
            inv_space_at_node = inv_space.gather(-1, gather_idx).squeeze(-1)  # [*B]

            # Maximum feasible delivery: limited by node space and vehicle capacity
            delivery = torch.min(inv_space_at_node, remaining_cap)  # [*B]
            delivery = delivery * is_customer.float()  # zero out depot visits

            # Scatter-add delivery into the per-period tally
            delivered = td["delivered"]  # [*B, N]
            delivered = delivered.scatter_add(-1, gather_idx, delivery.unsqueeze(-1))
            td["delivered"] = delivered

            # Consume vehicle capacity
            td["remaining_capacity"] = (remaining_cap - delivery).clamp(min=0.0)

        # ----------------------------------------------------------
        # Period transition
        # ----------------------------------------------------------
        if period_end.any():
            current_inv = td["current_inventory"]  # [*B, N]
            delivered = td["delivered"]  # [*B, N]
            demands = td["demands"]  # [*B, N]
            holding_costs = td["holding_costs"]  # [*B, N]

            # Inventory update: I_it = I_{i,t-1} + q_it - d_it
            new_inv = current_inv + delivered - demands  # [*B, N]

            # Holding cost for this period: sum_i h_i * max(I_it, 0)
            period_holding = (holding_costs * new_inv.clamp(min=0.0)).sum(-1)  # [*B]

            # Stockout: sum_i max(0, -I_it)  (unmet demand)
            period_stockout = (-new_inv).clamp(min=0.0).sum(-1)  # [*B]

            # Accumulate — only for instances where period_end is True
            pe_float = period_end.float()  # [*B]
            td["total_holding_cost"] = td["total_holding_cost"] + period_holding * pe_float
            td["total_stockout"] = td["total_stockout"] + period_stockout * pe_float

            # Carry inventory forward (clamp at 0; deficit is captured in stockout)
            new_inv_clamped = new_inv.clamp(min=0.0)
            pe_expand_n = period_end.unsqueeze(-1).expand_as(current_inv)
            td["current_inventory"] = torch.where(pe_expand_n, new_inv_clamped, current_inv)

            # Advance period counter
            td["current_period"] = td["current_period"] + period_end.long()

            # Reset vehicle capacity
            td["remaining_capacity"] = torch.where(
                period_end,
                td["vehicle_capacity"],
                td["remaining_capacity"],
            )

            # Reset delivered tally
            zero_delivered = torch.zeros_like(td["delivered"])
            pe_expand_n2 = period_end.unsqueeze(-1).expand_as(td["delivered"])
            td["delivered"] = torch.where(pe_expand_n2, zero_delivered, td["delivered"])

            # Reset visited mask: mark all False except depot (index 0)
            num_nodes = td["locs"].shape[-2]
            new_visited = torch.zeros(*td.batch_size, num_nodes, dtype=torch.bool, device=td.device)
            new_visited[..., 0] = True
            pe_expand_nodes = period_end.unsqueeze(-1).expand_as(td["visited"])
            td["visited"] = torch.where(pe_expand_nodes, new_visited, td["visited"])

        return td

    def _check_done(self, td: TensorDict) -> torch.Tensor:
        """
        Episode terminates when *current_period* reaches the planning horizon T.

        The condition fires immediately after the vehicle returns to the depot
        at the end of the last period, incrementing current_period to num_periods.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            BoolTensor [*B] — True for completed episodes.
        """
        done = td["current_period"] >= self.num_periods
        try:
            return done.reshape(td.batch_size)
        except Exception:
            return done.flatten().reshape(td.batch_size)

    def _get_action_mask(self, td: TensorDict) -> torch.Tensor:
        """
        Compute the boolean action mask for the current state.

        Valid actions
        -------------
        - Depot (index 0): always valid — allows the agent to end the current
          period's route at any time.
        - Customer i (index 1 … N): valid when ALL of the following hold:
            1. Not yet visited in the current period.
            2. The node's inventory is below its capacity (delivery > 0 possible).
            3. The vehicle has remaining capacity to deliver.

        A customer node where delivery would be zero (inventory full OR vehicle
        empty) is masked out to prevent wasteful detours.

        Args:
            td: Input TensorDict containing action and state information.

        Returns:
            BoolTensor [*B, num_nodes] — True indicates a selectable action.
        """
        # Base: unvisited nodes
        mask = ~td["visited"]  # [*B, N+1]

        # ---- Customer feasibility ----
        current_inv = td["current_inventory"]  # [*B, N]
        inv_cap = td["inventory_capacity"]  # [*B, N]
        has_inv_space = current_inv < inv_cap  # [*B, N]

        remaining_cap = td["remaining_capacity"]  # [*B]
        has_vehicle_cap = remaining_cap > 1e-7  # [*B]

        cust_mask = mask[..., 1:] & has_inv_space & has_vehicle_cap.unsqueeze(-1)
        mask[..., 1:] = cust_mask

        # ---- Depot always valid ----
        mask[..., 0] = True

        # ---- After planning horizon: no more customer visits ----
        done = td["current_period"] >= self.num_periods  # [*B]
        if done.any():
            done_expand = done.unsqueeze(-1).expand(*td.batch_size, mask.shape[-1] - 1)
            mask[..., 1:] = mask[..., 1:] & ~done_expand

        return mask

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the IRP episode reward (dense — meaningful at terminal step).

        reward = - routing_cost_weight  * total_routing_cost
                 - holding_cost_weight  * total_holding_cost
                 - stockout_penalty     * total_stockout

        *total_routing_cost* is the accumulated tour distance across all T
        periods plus the final return-to-depot distance if the vehicle ends a
        period at a customer node (edge case guard; should not occur in normal
        play since the agent must return to depot to advance periods).

        Cost components are stored back into the TensorDict for downstream
        logging (``routing_cost``, ``holding_cost``, ``stockout``).

        Args:
            td: Input TensorDict containing action and state information.
            actions: Action sequence (unused in this implementation).

        Returns:
            Tensor [*B] — scalar reward per batch instance.
        """
        # ---- Routing cost ----
        routing_cost = td["tour_length"].clone()  # [*B]

        # Guard: if episode ends while not at depot, add the residual return dist
        current = td["current_node"]
        if current.dim() > len(td.batch_size):
            current = current.squeeze(-1)

        locs = td["locs"]
        # Gather current position coordinates
        gather_loc = current.view(*td.batch_size, 1, 1).expand(*td.batch_size, 1, 2)
        current_loc = locs.gather(-2, gather_loc).squeeze(-2)  # [*B, 2]
        depot_loc = td["depot"]  # [*B, 2]
        return_dist = torch.norm(depot_loc - current_loc, dim=-1)  # [*B]
        not_at_depot = (current != 0).float()
        routing_cost = routing_cost + return_dist * not_at_depot

        # ---- Holding and stockout costs ----
        holding_cost = td["total_holding_cost"]  # [*B]
        stockout = td["total_stockout"]  # [*B]

        # ---- Store components for logging ----
        td["routing_cost"] = routing_cost
        td["holding_cost"] = holding_cost
        td["stockout"] = stockout

        # ---- Scalar reward ----
        reward = (
            -self.routing_cost_weight * routing_cost
            - self.holding_cost_weight * holding_cost
            - self.stockout_penalty * stockout
        )

        if reward.dim() > len(td.batch_size):
            reward = reward.squeeze(-1)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)

        return reward.view(td.batch_size)


class IRP(BaseProblem):
    """Inventory Routing Problem (IRP) task definition."""

    NAME = "irp"

    @staticmethod
    def evaluate_tour(
        dataset: Dict[str, Any],
        pi: torch.Tensor,
        dist_matrix: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute IRP costs and outcomes."""
        routing_cost = dataset.get("routing_cost")
        if routing_cost is not None and isinstance(routing_cost, torch.Tensor):
            holding_cost = dataset.get("holding_cost", torch.zeros_like(routing_cost))
            stockout = dataset.get("stockout", torch.zeros_like(routing_cost))
            cost = routing_cost + holding_cost - stockout
        else:
            length = IRP.get_tour_length(dataset, pi, dist_matrix)
            cost = length
        return {
            "cost": cost,
            "reward": -cost,
            "length": cost,
            "valid": torch.ones_like(cost, dtype=torch.bool),
            "total": cost,
        }


IRP_SPEC = ProblemSpec(
    name="irp",
    objective=lambda cost, **kw: -float(cost),
    constraints=(),
    metrics=("cost", "reward", "length"),
    env_target="logic.src.pipeline.rl.envs.irp:IRPEnv",
    expand_pool=False,
)
PROBLEMS.register(IRP_SPEC)

__all__ = ["IRPGenerator", "IRPEnv", "IRP", "IRP_SPEC"]
