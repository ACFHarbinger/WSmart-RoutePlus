"""
PTP Environment implementation.

Profitable Tour Problem: Select profitable subset
of nodes to visit while minimizing travel cost.

Attributes:
    NAME: Environment identifier string ``"ptp"``.
    name: Alias for NAME used by the registry.
    node_dim: Number of node features (x, y, waste, profit) = 4.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("ptp", num_loc=50)
    >>> td = env.reset()
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.generators import PTPGenerator


class PTPEnv(RL4COEnvBase):
    """
    Profitable Tour Problem Environment.

    The agent must select which nodes to visit to maximize
    total waste collected minus travel cost.

    Attributes:
        NAME: Environment identifier string ``"ptp"``.
        name: Alias for NAME used by the registry.
        node_dim: Number of node features (x, y, waste, profit) = 4.
    """

    NAME = "ptp"
    name: str = "ptp"

    def __init__(
        self,
        generator: Optional[PTPGenerator] = None,
        generator_params: Optional[dict] = None,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        revenue_kg: Optional[float] = None,
        cost_km: Optional[float] = None,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """
        Initialize PTPEnv.

        Args:
            generator: Problem instance generator.
            generator_params: Parameters for generator initialization.
            waste_weight: Weight for waste collection in reward.
            cost_weight: Weight for travel cost in reward.
            revenue_kg: Optional revenue per kg (overrides waste_weight).
            cost_km: Optional cost per km (overrides cost_weight).
            device: Device for torch tensors ('cpu' or 'cuda').
            kwargs: Additional keyword arguments.
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = PTPGenerator(**generator_params, device=device)

        super().__init__(generator, generator_params, device, **kwargs)
        self.waste_weight = revenue_kg if revenue_kg is not None else waste_weight
        self.cost_weight = cost_km if cost_km is not None else cost_weight

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        """
        This method sets up the initial state for a Profitable Tour Problem (PTP)
        episode. It handles depot prepending logic to ensure coordinates are in the correct format
        (N+1 nodes including depot at index 0), and initializes all tracking fields.

        State Initialization Logic:
            1. Check if depot needs to be prepended to customer locations
            2. Prepend depot to both locations and waste arrays if needed
            3. Initialize tracking fields: visited, tour, tour_length, collected_waste

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - locs (Tensor): Location coordinates, shape (batch, N, 2) or (batch, N+1, 2)
                - depot (Tensor): Depot coordinates, shape (batch, 2)
                - waste (Tensor, optional): Waste at each location, shape (batch, N) or (batch, N+1)
                - visited (Tensor, optional): If present, skip reinitialization (transductive search)

        Returns:
            TensorDict: Updated state dictionary with added fields:
                - current_node (LongTensor): Current position index, shape (batch, 1), initialized to 0 (depot)
                - visited (BoolTensor): Visit status for each node, shape (batch, num_nodes), depot marked visited
                - tour (LongTensor): Sequence of visited node indices, shape (batch, 0), starts empty
                - tour_length (Tensor): Total distance traveled, shape (batch,), initialized to 0.0
                - collected_waste (Tensor): Total waste/profit collected, shape (batch,), initialized to 0.0

        Note:
            The robust depot prepending logic handles three scenarios:
            1. Generator provides num_loc - compare actual shape to expected N+1
            2. No generator hint - check if first location matches depot coordinates
            3. Mismatched waste array - ensure waste has same node count as locations
        """
        if self.generator is None:
            raise ValueError(f"Generator for {self.NAME} is not initialized. Initialize with an instance first.")

        # If state is already initialized (e.g. transductive search resuming), skip coord mod
        if "visited" in tensordict.keys():
            return tensordict

        device = tensordict.device
        bs = tensordict.batch_size

        # Prepend depot to locs and waste if they are customer-only
        locs = tensordict["locs"]
        waste = tensordict.get("waste")

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
            raise ValueError("Generator must be initialized for PTP environment.")
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

        # Initialize state fields
        tensordict["current_node"] = torch.zeros(*bs, 1, dtype=torch.long, device=device)
        tensordict["visited"] = torch.zeros(*bs, num_nodes, dtype=torch.bool, device=device)
        tensordict["visited"].index_fill_(-1, torch.tensor([0], device=device), True)  # Depot index 0

        # Tour tracking
        tensordict["tour"] = torch.zeros(*bs, 0, dtype=torch.long, device=device)
        tensordict["tour_length"] = torch.zeros(*bs, device=device)
        tensordict["collected_waste"] = torch.zeros(*bs, device=device)
        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute action and update state for PTP.

        Args:
            tensordict: Current state TensorDict containing ``action``, ``current_node``,
                ``visited``, ``tour``, ``tour_length``, ``collected_waste``,
                ``locs``, ``depot``, ``waste``, ``revenue_kg``, and ``cost_km`` fields.

        Returns:
            TensorDict: Updated state with ``current_node``, ``visited``,
                ``tour``, ``tour_length``, ``collected_waste``, ``reward``,
                ``terminated``, and ``truncated`` fields.
        """
        return OpsMixin._step(self, tensordict)

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
        """
        Execute a routing action and update the episode state.

        Delegates common routing mechanics (distance, visited, current_node, tour)
        to OpsMixin._step_instance, then applies PTP-specific waste collection.

        Args:
            tensordict (TensorDict): Current state containing:
                - action (LongTensor): Selected next node index, shape (batch,)
                - current_node (LongTensor): Current position, shape (batch, 1)
                - locs (Tensor): All node coordinates including depot, shape (batch, num_nodes, 2)
                - waste (Tensor): Waste/profit at each node, shape (batch, num_nodes)
                - visited (BoolTensor): Visit status, shape (batch, num_nodes)
                - tour (LongTensor): Historical tour, shape (batch, tour_len)
                - tour_length (Tensor): Cumulative distance, shape (batch,)
                - collected_waste (Tensor): Cumulative waste collected, shape (batch,)

        Returns:
            TensorDict: Updated state with modifications to:
                - tour_length: Incremented by Euclidean distance
                - collected_waste: Incremented by waste[action] if valid collection
                - visited: action node marked as True
                - current_node: Updated to action value
                - tour: Action appended to tour sequence

        Note:
            The method handles edge cases where waste array may not include depot
            (shape mismatch) by dynamically prepending a zero depot waste value.
        """
        action = tensordict["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)
        if action.dim() == 0:
            action = action.unsqueeze(0)

        # Capture new-visit status before super() updates the visited mask
        is_new_visit = ~tensordict["visited"].gather(1, action.unsqueeze(-1)).squeeze(-1)
        is_not_depot = action != 0

        # Delegate distance / visited / current_node / tour updates
        tensordict = super()._step_instance(tensordict)

        bs = tensordict.batch_size
        device = tensordict.device

        # Waste collection (PTP-specific)
        waste = tensordict.get("waste")
        if waste is not None and waste.shape[-1] == tensordict["visited"].shape[-1] - 1:
            waste = torch.cat([torch.zeros(*bs, 1, device=device), waste], dim=1)
            tensordict["waste"] = waste

        if waste is not None:
            waste_collected = waste.gather(1, action.unsqueeze(-1)).squeeze(-1)
            tensordict["collected_waste"] = (
                tensordict["collected_waste"] + waste_collected * is_new_visit.float() * is_not_depot.float()
            )

        return tensordict

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Compute action mask for PTP with PTP-correct termination logic.

        Standard behavior:
        - Mask out already-visited nodes.
        - Mask out bins with zero waste (nothing to collect).
        - Depot (index 0) is **always legal** so the agent can terminate early.

        Mandatory behavior (unchanged):
        - If pending mandatory bins remain, depot is blocked until they are served.
        - Once all mandatory bins are visited, depot becomes legal again.

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - visited (BoolTensor): Visit status for each node, shape (batch, num_nodes)
                - waste (Tensor, optional): Waste at each node, shape (batch, num_nodes)
                - mandatory (BoolTensor, optional): Mandatory constraints, shape (batch, num_nodes)

        Returns:
            Tensor: Boolean mask (batch, num_nodes) where True = valid action.
        """
        visited = tensordict["visited"]  # [B, N]
        waste = tensordict.get("waste", None)  # [B, N] or None

        # 1. Exclude already-visited nodes
        mask = ~visited.clone()

        # 2. Exclude bins with no waste (nothing to collect).
        #    Depot waste is 0 after prepend; we restore it below.
        if waste is not None:
            # Guard: if waste has fewer nodes than visited (no depot col), don't apply to depot
            if waste.shape[-1] == visited.shape[-1]:
                mask = mask & (waste > 0)
            else:
                # waste doesn't include depot column — apply only to customer columns
                mask[:, 1:] = mask[:, 1:] & (waste > 0)

        # 3. Depot is ALWAYS legal — agent can return and terminate at any time
        mask[:, 0] = True

        # 4. Mandatory routing override
        mandatory = tensordict.get("mandatory", None)
        if mandatory is not None:
            # Collapse any unexpected extra dimensions so mandatory is 2D [B, N].
            # This guards against stale 3D masks from pre-reset state.
            loop_count = 0
            while mandatory.dim() > 2:
                mandatory = mandatory.any(dim=1)
                loop_count += 1
                if loop_count > 10:
                    break

            if mandatory.shape[-1] == visited.shape[-1] - 1:
                depot_mandatory = torch.zeros(mandatory.shape[0], 1, dtype=torch.bool, device=mandatory.device)
                mandatory = torch.cat([depot_mandatory, mandatory], dim=1)

            # Pending mandatory bins: mandatory AND not yet visited
            pending_mandatory = mandatory & ~visited

            # Mandatory bins are a hard operational constraint that overrides the
            # waste > 0 filter. A bin with zero current fill can still be mandatory
            # (e.g. recently collected but high accumulation rate means it will
            # overflow before the next collection cycle). Without this override,
            # the decoder deadlocks: the waste filter blocks the bin, mandatory
            # blocks the depot, and all logits become -inf → NaN.
            mask[:, 1:] = mask[:, 1:] | pending_mandatory[:, 1:]

            # Check if any mandatory bins remain (excluding depot at index 0).
            # Result is guaranteed 1D [B] after .any(dim=1) on a 2D tensor.
            has_pending_mandatory = pending_mandatory[:, 1:].any(dim=1)

            # Block depot while mandatory bins are outstanding
            mask[:, 0] = ~has_pending_mandatory

        return mask

    def _get_reward(self, tensordict: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute PTP reward: waste - cost.

        Reward = (waste_weight * total_waste) - (cost_weight * tour_length)

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - collected_waste (Tensor): Total waste collected, shape (batch,)
                - tour_length (Tensor): Total distance traveled, shape (batch,)
                - current_node (LongTensor): Current position index, shape (batch, 1)
                - locs (Tensor): Location coordinates, shape (batch, num_nodes, 2)
                - depot (Tensor): Depot coordinates, shape (batch, 2)
                - visited (BoolTensor): Visit status for each node, shape (batch, num_nodes)
            actions (Optional[torch.Tensor]): Optional tour action sequence ``(batch, tour_len)`` (unused).

        Returns:
            Tensor: Reward for each batch, shape (batch,)
        """
        waste = tensordict.get("collected_waste", torch.zeros_like(tensordict["tour_length"]))
        cost = tensordict["tour_length"]

        # Also add return to depot distance if not already there
        current = tensordict["current_node"].squeeze(-1)
        # Combine depot and customers for full locs if not already done
        locs = tensordict["locs"]
        if locs.shape[-2] == tensordict.get("visited", torch.zeros(0)).shape[-1] - 1:
            locs = torch.cat([tensordict["depot"].unsqueeze(1), locs], dim=1)

        current_idx = current[:, None, None].expand(-1, -1, 2)
        current_loc = locs.gather(1, current_idx).squeeze(1)
        depot_loc = tensordict["depot"]

        dm = tensordict.get("dm", None)
        if dm is not None:
            # Road distance from current node back to depot (index 0)
            return_distance = dm.gather(1, current[:, None, None].expand(-1, -1, dm.shape[-1])).squeeze(1)[:, 0]
        else:
            return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        # Only add return distance if not already at depot
        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Store decomposed rewards for GDPO
        tensordict["reward_waste"] = waste
        tensordict["reward_cost"] = -total_cost  # Convention: maximized, so negative cost

        reward = self.waste_weight * waste - self.cost_weight * total_cost
        return reward

    def _check_done(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Check if the episode is done.

        Args:
            tensordict (TensorDict): Input tensor dictionary containing:
                - current_node (LongTensor): Current position index, shape (batch, 1)
                - i (LongTensor): Current step, shape (batch, 1)

        Returns:
            Tensor: Boolean tensor indicating if the episode is done, shape (batch,)
        """
        current = tensordict["current_node"].squeeze(-1)
        step = tensordict["i"].squeeze(-1)
        return (current == 0) & (step > 0)
