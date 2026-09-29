"""
WCVRP Environment implementation.

Waste Collection Vehicle Routing Problem: Collect waste from bins
while respecting vehicle capacity constraints.

Attributes:
    WCVRPEnv: Class definition of WCVRP Env.

Example:
    >>> from logic.src.envs.generators import WCVRPGenerator
    >>> generator = WCVRPGenerator(num_nodes=10, generator_params={"num_nodes": 10})
    >>> env = WCVRPEnv(generator)
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.ops import OpsMixin, as_batch_nodes
from logic.src.envs.generators import WCVRPGenerator


class WCVRPEnv(RL4COEnvBase):
    """
    Waste Collection VRP Environment.

    The agent must collect waste from bins before they overflow,
    while minimizing travel cost and respecting vehicle capacity.

    Attributes:
        WCVRPEnv: Class definition of WCVRP Env.
    """

    NAME = "wcvrp"
    name: str = "wcvrp"

    def __init__(
        self,
        generator: Optional[WCVRPGenerator] = None,
        generator_params: Optional[dict] = None,
        overflow_penalty: float = 10.0,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """
        Initialize WCVRPEnv.



        Args:
            generator: Problem instance generator.
            generator_params: Parameters for generator initialization.
            overflow_penalty: Penalty for bin overflow.
            waste_weight: Reward weight for waste collection.
            cost_weight: Weight for travel cost in reward.
            revenue_kg: Optional revenue per kg (overrides waste_weight).
            cost_km: Optional cost per kg (overrides cost_weight).
            device: Device for torch tensors ('cpu' or 'cuda').
            kwargs: Additional keyword arguments.
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = WCVRPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)
        self.overflow_penalty = overflow_penalty
        self.waste_weight = revenue_kg if revenue_kg is not None else waste_weight
        self.cost_weight = cost_km if cost_km is not None else cost_weight

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        """Initialize Waste Collection VRP (WCVRP) episode state with capacity tracking.

        This method extends the base VRPP reset logic to add capacity constraints and overflow
        monitoring for waste collection scenarios. It initializes bin fill levels, vehicle load
        tracking, and overflow counting.

        Capacity-Specific Initialization:
            - current_load: Vehicle's current waste load (starts at 0)
            - total_collected: Cumulative waste collected across all depot returns
            - cur_overflows: Count of bins currently at or exceeding max_waste threshold

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - locs (Tensor): Location coordinates, shape (batch, N, 2) or (batch, N+1, 2)
                - depot (Tensor): Depot coordinates, shape (batch, 2)
                - waste (Tensor): Current bin fill levels, shape (batch, N) or (batch, N+1)
                - max_waste (Tensor): Maximum bin capacity per node, shape (batch, 1) or (batch, N)
                - capacity (Tensor): Vehicle capacity, shape (batch,)
                - visited (Tensor, optional): If present, skip reinitialization

        Returns:
            TensorDict: Updated state with WCVRP-specific fields:
                - current_node (LongTensor): Starting at depot (0), shape (batch,)
                - visited (BoolTensor): Depot marked as visited, shape (batch, num_nodes)
                - tour (LongTensor): Empty tour sequence, shape (batch, 0)
                - tour_length (Tensor): Zero distance, shape (batch,)
                - current_load (Tensor): Zero initial load, shape (batch,)
                - collected_waste (Tensor): Zero collected, shape (batch,)
                - total_collected (Tensor): Zero total, shape (batch,)
                - cur_overflows (Tensor): Count of overflowing bins (waste >= max_waste), shape (batch,)

        Note:
            Overflow counting excludes the depot (index 0) and only considers customer bins.
            The initial overflow count is computed as sum(waste[1:] >= max_waste[1:]).
        """
        if self.generator is None:
            raise ValueError(f"Generator for {self.NAME} is not initialized. Initialize with an instance first.")

        # If state already initialized (e.g. transductive search resuming), skip coord mod
        if "visited" in tensordict.keys():
            return tensordict

        device = tensordict.device
        bs = tensordict.batch_size

        # Prepend depot to locs and waste if they are customer-only
        locs = tensordict["locs"]
        waste = tensordict["waste"]

        # Temporal datasets store waste as 3D (batch, num_days, num_loc).
        # apply_time_step() converts it to 2D after epoch 0, but on epoch 0 we
        # must extract the current day's slice before any depot-prepend logic.
        if waste is not None and waste.dim() > 2:
            current_day_t = tensordict.get("current_day", None)
            day_idx = int(current_day_t.reshape(-1)[0].item()) if current_day_t is not None else 0
            day_idx = min(day_idx, waste.shape[1] - 1)
            waste = waste[:, day_idx, :]
            tensordict["waste"] = waste

        # Robust N vs N+1 logic
        # gen_n represents customers only. If shape is gen_n + 1, it already has depot.
        if self.generator is None:
            raise ValueError("Generator must be initialized for WCVRP environment.")
        gen_n = self.generator.num_loc

        needs_prepend = False
        if "depot" in tensordict.keys():
            # 1. If we know N, check if we are already at N+1
            if gen_n is not None and locs.shape[-2] == gen_n + 1:
                needs_prepend = False
            # 2. If we are at N, we definitely need to prepend
            elif (
                gen_n is not None
                and locs.shape[-2] == gen_n
                or not torch.allclose(locs[..., 0, :], tensordict["depot"], atol=1e-4)
            ):
                needs_prepend = True

        if needs_prepend:
            tensordict["locs"] = torch.cat([tensordict["depot"].unsqueeze(1), locs], dim=1)
            locs = tensordict["locs"]
            if waste is not None:
                tensordict["waste"] = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)
        elif waste is not None and waste.shape[-1] == locs.shape[-2] - 1:
            # Catch case where locs has depot but waste doesn't
            tensordict["waste"] = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)

        num_nodes = tensordict["locs"].shape[-2]

        tensordict["current_node"] = torch.zeros(bs, dtype=torch.long, device=device)
        tensordict["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        tensordict["visited"].index_fill_(-1, torch.tensor([0], device=device), True)

        tensordict["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        tensordict["tour_length"] = torch.zeros(*bs, device=device)
        tensordict["current_load"] = torch.zeros(*bs, device=device)
        tensordict["collected_waste"] = torch.zeros(*bs, device=device)
        tensordict["total_collected"] = torch.zeros(*bs, device=device)

        # Calculate initial overflows (consistent with _get_reward: count of nodes >= max_waste)
        max_waste = tensordict.get("max_waste", torch.tensor(1.0, device=device))
        mw = max_waste.unsqueeze(-1) if max_waste.dim() == 1 else max_waste

        # Customers only (skip depot at index 0)
        waste_cust = tensordict["waste"][..., 1:]
        max_w_cust = mw if mw.shape[-1] == 1 else mw[..., 1:]

        tensordict["cur_overflows"] = (waste_cust >= max_w_cust).float().sum(-1)

        return tensordict

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
        """
        Execute a waste collection action with capacity constraints and bin clearing.

        This method extends the base VRPP step logic to handle:
        1. Capacity-constrained waste collection (collect min of bin level and vehicle capacity)
        2. Depot emptying (vehicle load resets to 0 when returning to depot)
        3. Bin clearing (collected bins have waste set to 0)
        4. Load tracking across multiple depot visits

        Collection Mechanics:
            - At customer nodes (action != 0):
                * Collect min(waste_at_node, max_waste, remaining_capacity)
                * Add collected amount to current_load and total_collected
                * Set bin waste level to 0 after collection
            - At depot (action == 0):
                * Reset current_load to 0 (emptying vehicle)
                * total_collected remains unchanged (cumulative across trips)

        Args:
            tensordict (TensorDict): Current state containing:
                - action (LongTensor): Selected node to visit, shape (batch,)
                - current_node (LongTensor): Current position, shape (batch,)
                - current_load (Tensor): Current vehicle load, shape (batch,)
                - waste (Tensor): Bin fill levels, shape (batch, num_nodes)
                - max_waste (Tensor): Bin capacity limits, shape (batch, 1) or (batch, num_nodes)
                - capacity (Tensor): Vehicle capacity, shape (batch,)
                - visited, tour, tour_length: Inherited tracking fields

        Returns:
            TensorDict: Updated state with modifications to:
                - current_load: Increased by collected waste, or reset to 0 at depot
                - collected_waste: Incremented by collection amount
                - total_collected: Cumulative waste across all trips (equals collected_waste)
                - waste: Bin at action index set to 0 after collection
                - current_node: Updated to action value
                - tour, tour_length, visited: Updated via parent class

        Note:
            The method calls super()._step_instance(tensordict) first to handle base VRP state updates
            (distance, tour, visited), then applies capacity-specific waste collection logic.
        """
        action = as_batch_nodes(tensordict["action"])
        tensordict["action"] = action

        # Core mechanics
        tensordict = super()._step_instance(tensordict)

        is_not_depot = action != 0
        waste = tensordict["waste"]
        waste_at_node = waste.gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Collect waste (clamped to max_waste and remaining vehicle capacity)
        max_w = tensordict.get("max_waste", torch.tensor(1e9, device=tensordict.device))
        if max_w.dim() > 1 and max_w.shape[-1] == tensordict["visited"].shape[-1] - 1:
            max_w = torch.cat(
                [torch.tensor(1e9, device=tensordict.device).expand(*tensordict.batch_size, 1), max_w], dim=1
            )
        max_w_at_node = max_w.gather(1, action.unsqueeze(-1)).squeeze(-1) if max_w.dim() > 1 else max_w

        collected = torch.min(waste_at_node, max_w_at_node) * is_not_depot.float()
        tensordict["current_load"] = tensordict["current_load"] + collected
        tensordict["collected_waste"] = tensordict["collected_waste"] + collected
        tensordict["total_collected"] = tensordict["collected_waste"]  # Alias

        # Empty at depot
        at_depot = action == 0
        tensordict["current_load"] = torch.where(
            at_depot, torch.zeros_like(tensordict["current_load"]), tensordict["current_load"]
        )

        # Visiting a bin empties it. Overflow credit is still clamped to
        # max_waste above; the residual fill is discarded so a served bin
        # cannot keep counting as an overflow.
        tensordict["waste"].scatter_(1, action.unsqueeze(-1), 0)

        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Step through environment with OpsMixin.

        Args:
            tensordict: TensorDict containing the state.

        Returns:
            TensorDict: The next state.
        """
        return OpsMixin._step(self, tensordict)

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Compute action mask based on capacity, visited status, and mandatory constraints.

        This method implements three-tier filtering to determine valid actions:
        1. **Visit filter**: Exclude already-visited nodes (except depot)
        2. **Capacity filter**: Exclude bins exceeding remaining vehicle capacity
        3. **Mandatory filter**: Enforce priority routing to critical bins

        Mandatory Routing Logic:
            - If mandatory is None: Standard VRP behavior, depot always valid
            - If mandatory has True values: MUST visit those bins first; depot blocked until complete
            - If mandatory is all False: Episode complete; only depot is valid (return home)

        Edge Cases:
            - **Vehicle full with mandatory bins**: Depot becomes valid to allow emptying
            - **All mandatory bins visited**: Depot becomes valid for return
            - **No valid customer nodes**: Depot is always fallback to prevent deadlock
            - **Waste array mismatch**: Dynamically prepends depot zero if needed

        Args:
            tensordict (TensorDict): Current state containing:
                - visited (BoolTensor): Visit status, shape (batch, num_nodes)
                - waste (Tensor): Bin fill levels, shape (batch, num_nodes) or (batch, N)
                - current_load (Tensor): Current vehicle load, shape (batch,)
                - capacity (Tensor): Vehicle capacity, shape (batch,)
                - mandatory (BoolTensor, optional): Priority bins, shape (batch, num_nodes)

        Returns:
            Tensor: Boolean action mask (batch, num_nodes) where True indicates a valid action.
                    At least one action per batch instance is guaranteed to be valid.

        Note:
            The capacity check uses remaining_capacity = capacity - current_load.
            Nodes requiring more waste collection than remaining capacity are masked out,
            forcing a depot return when the vehicle is near full.
        """
        mask = ~tensordict["visited"].clone()

        # Check capacity constraints
        waste = tensordict["waste"]
        if waste.shape[-1] == mask.shape[-1] - 1:
            waste = torch.cat([torch.zeros(*tensordict.batch_size, 1, device=tensordict.device), waste], dim=1)

        remaining_capacity = tensordict["capacity"].unsqueeze(-1) - tensordict["current_load"].unsqueeze(-1)
        exceeds_capacity = waste > remaining_capacity

        mask = mask & ~exceeds_capacity

        # Mandatory routing logic
        mandatory = tensordict.get("mandatory", None)

        if mandatory is not None:
            # mandatory: (batch, num_nodes) boolean tensor
            # True = must visit this bin, False = optional
            if mandatory.size(-1) == mask.size(-1) - 1:
                mandatory = torch.cat([torch.zeros_like(mandatory[:, :1], dtype=torch.bool), mandatory], dim=1)

            # Pending mandatory bins: mandatory AND not yet visited AND within capacity
            pending_mandatory = mandatory & mask

            # Check if any mandatory bins remain (excluding depot at index 0)
            has_pending_mandatory = pending_mandatory[:, 1:].any(dim=1)

            # Depot is valid only if no pending mandatory bins remain
            # If has_pending_mandatory is True -> depot invalid (False)
            # If has_pending_mandatory is False -> depot valid (True)
            mask[:, 0] = ~has_pending_mandatory
        else:
            # No mandatory constraint: depot always valid
            mask[:, 0] = True

        return mask

    def _get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute WCVRP reward.

        Efficient routing (low cost, high collection) = high reward.



        Args:
            tensordict (TensorDict): Current state containing:
                - total_collected (Tensor): Total waste collected, shape (batch,)
                - tour_length (Tensor): Current tour length, shape (batch,)
                - current_node (Tensor): Current node index, shape (batch,)
                - locs (Tensor): Node locations, shape (batch, num_nodes, 2)
                - depot (Tensor): Depot location, shape (batch, 2)
                - visited (BoolTensor): Visit status, shape (batch, num_nodes)
                - waste (Tensor): Bin fill levels, shape (batch, num_nodes) or (batch, N)
                - max_waste (Tensor): Maximum bin capacity, shape (batch,) or (batch, N)
            actions (Optional[torch.Tensor]): Actions taken (optional, not used in this reward calculation).

        Returns:
            torch.Tensor: Reward tensor, shape (batch,)

        """
        collection = tensordict["total_collected"]
        cost = tensordict["tour_length"]

        # Add return to depot
        current = tensordict["current_node"]
        if current.dim() > 1:
            current = current.squeeze(-1)
        if current.dim() == 0:
            current = current.unsqueeze(0)

        locs = tensordict["locs"]
        # Use robust concatenation
        if locs.shape[-2] == tensordict.get("visited", torch.zeros(0)).shape[-1] - 1:
            locs = torch.cat([tensordict["depot"].unsqueeze(1), locs], dim=1)

        # Use robust indexing
        current_idx = current[:, None, None].expand(-1, -1, 2)
        current_loc = locs.gather(1, current_idx).squeeze(1)
        depot_loc = tensordict["depot"]

        dm = tensordict.get("dm", None)
        if dm is not None:
            # Road distance from current node back to depot (index 0)
            return_distance = dm.gather(1, current[:, None, None].expand(-1, -1, dm.shape[-1])).squeeze(1)[:, 0]
        else:
            return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Calculate overflows: unvisited nodes where waste >= max_waste
        # In current WCVRP, visited nodes have waste=0 in the final td.
        max_waste = tensordict.get("max_waste", torch.tensor(1.0, device=tensordict.device))

        # We only care about customers (index 1 onwards) for overflows
        waste = tensordict["waste"][..., 1:]
        if max_waste.dim() > 1:  # (B, N)
            max_waste = max_waste[..., 1:]
        elif max_waste.dim() == 1:
            max_waste = max_waste.unsqueeze(-1)

        overflows = (waste >= max_waste).float().sum(-1)

        # Store individual components in TensorDict for logging/meta access
        tensordict["collection"] = collection
        tensordict["cost"] = total_cost
        tensordict["overflows"] = overflows

        # Store decomposed rewards for GDPO (signed for maximization)
        tensordict["reward_waste"] = collection
        tensordict["reward_cost"] = -total_cost
        tensordict["reward_overflow"] = -overflows

        reward = self.waste_weight * collection - self.cost_weight * total_cost - self.overflow_penalty * overflows

        # Ensure it's 1D [B] matching batch_size
        if reward.dim() > 1:
            reward = reward.squeeze(-1)
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)

        return reward.view(tensordict.batch_size)
