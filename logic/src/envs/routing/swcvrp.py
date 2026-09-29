"""
SCWCVRP Environment implementation.

Stochastic Capacitated Waste Collection Vehicle Routing Problem:
Agent sees noisy estimates of bin fill levels but collects real waste.
Rewards are based on real waste levels and overflows.

Attributes:
    NAME: Environment identifier string ``"swcvrp"``.
    name: Alias for NAME used by the registry.
    node_dim: Node feature dimension — 2 for (x, y) coordinates.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("swcvrp", num_loc=20)
    >>> td = env.reset(batch_size=[4])
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase

from logic.src.envs.base.ops import as_batch_nodes
from logic.src.envs.generators import SCWCVRPGenerator
from logic.src.envs.routing.wcvrp import WCVRPEnv


class SCWCVRPEnv(WCVRPEnv):
    """
    Stochastic Capacitated Waste Collection VRP Environment.

    The agent must collect waste from bins based on noisy observations.
    The real waste level is used for reward calculation (overflows).

    Attributes:
        name: Alias for NAME used by the registry.
        generator_params: Keyword arguments forwarded to SCWCVRPGenerator constructor.
    """

    name: str = "scwcvrp"

    def __init__(
        self,
        generator: Optional[SCWCVRPGenerator] = None,
        generator_params: Optional[dict] = None,
        overflow_penalty: float = 10.0,
        waste_weight: float = 1.0,
        cost_weight: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        """Initialize SCWCVRPEnv.

        Args:
            generator: Data generator for the environment.
            generator_params: Parameters for the data generator.
            overflow_penalty: Description of overflow_penalty.
            waste_weight: Description of waste_weight.
            cost_weight: Description of cost_weight.
            device: Torch device to place tensors on.
            kwargs: Additional keyword arguments.
        """
        generator_params = generator_params or kwargs
        if generator is None:
            generator = SCWCVRPGenerator(**generator_params, device=device)

        super().__init__(
            generator=generator,
            generator_params=generator_params,
            overflow_penalty=overflow_penalty,
            waste_weight=waste_weight,
            cost_weight=cost_weight,
            device=device,
            **kwargs,
        )

    def _reset_instance(self, td: TensorDict) -> TensorDict:
        """Initialize SWCVRP episode state.

        Args:
            td: TensorDict produced by the generator, containing ``locs``
                (customer coordinates) and ``depot`` tensors.

        Returns:
            TensorDict: Initialised state with ``current_node``, ``visited``,
                ``tour``, ``tour_length``, ``first_node``, ``reward``,
                ``terminated``, and ``truncated`` fields.
        """
        if self.generator is None:
            raise ValueError(f"Generator for {self.name} is not initialized. Initialize with an instance first.")
        td = super()._reset_instance(td)

        # Ensure real_waste is tracked and has correct shape (N+1)
        if "real_waste" not in td.keys():
            # If not provided by generator, assume it's same as waste (no noise)
            td["real_waste"] = td["waste"].clone()
        else:
            # If provided by generator, it might be customer-only (N)
            # Prepend depot (0.0) if needed to match observed waste (N+1)
            real_waste = td["real_waste"]
            if real_waste.shape[-1] == td["waste"].shape[-1] - 1:
                bs = td.batch_size
                depot_waste = torch.zeros(*bs, 1, device=td.device)
                td["real_waste"] = torch.cat([depot_waste, real_waste], dim=1)

        # Track total real waste collected
        td["total_real_collected"] = torch.zeros_like(td["collected_waste"])

        return td

    def _step_instance(self, td: TensorDict) -> TensorDict:
        """Execute action and update state with real waste collection.

        Args:
            td: Current state containing ``action``, ``current_node``,
                ``capacity``, ``current_load``, ``real_waste``, ``locs``,
                and ``depot`` tensors.

        Returns:
            TensorDict: Updated state with incremented ``tour_length``, updated
                ``visited`` mask, ``current_node`` set to the selected action,
                and the action appended to ``tour``.
        """
        action = as_batch_nodes(td["action"])
        td["action"] = action
        is_not_depot = action != 0

        # Collection logic for real waste
        real_waste_at_node = td["real_waste"].gather(1, action.unsqueeze(-1)).squeeze(-1)

        # Update real waste tracking (internal)
        remaining_cap = td["capacity"] - td["current_load"]
        actual_collected = torch.min(real_waste_at_node, remaining_cap)
        actual_collected = actual_collected * is_not_depot.float()

        # Update real waste in the bins
        new_real_waste = td["real_waste"].clone()
        new_real_waste.scatter_(
            1, action.unsqueeze(-1), td["real_waste"].gather(1, action.unsqueeze(-1)) - actual_collected.unsqueeze(-1)
        )
        td["real_waste"] = new_real_waste

        # Now update noisy waste - usually setting to 0 after visit if we think we cleared it
        # or update based on what we think we collected?
        # Standard WCVRPEnv sets td['waste'] to 0 at the node.

        td = super()._step_instance(td)

        # Override the collection tracking with real values if desired,
        # or just track total_real_collected.
        td["total_real_collected"] = td["total_real_collected"] + actual_collected

        return td

    def _get_reward(self, td: TensorDictBase, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute SCWCVRP reward based on REAL waste levels and collection.
        Args:
            td: Final state TensorDict containing ``total_real_collected``,
                ``tour_length``, ``current_node``, ``locs``, and ``depot``.
            actions: Optional tour action sequence ``(batch, tour_len)`` (unused).

        Returns:
            torch.Tensor: Scalar reward per batch element, shape ``(batch,)``.
        """
        # We use total_real_collected instead of total_collected
        collection = td["total_real_collected"]
        cost = td["tour_length"]

        # Add return to depot
        current = td["current_node"].squeeze(-1)
        locs = td["locs"]
        current_loc = locs.gather(1, current[:, None, None].expand(-1, -1, 2)).squeeze(1)
        depot_loc = td["depot"]
        return_distance = torch.norm(depot_loc - current_loc, dim=-1)

        not_at_depot = current != 0
        total_cost = cost + return_distance * not_at_depot.float()

        # Overflows based on remaining REAL waste (customers only).
        # Do not blindly unsqueeze: a per-node [B, N+1] tensor would become
        # [B, N+1, 1] and broadcast against [B, N] incorrectly.
        max_waste = td.get("max_waste", torch.tensor(1.0, device=td.device))
        real_waste = td["real_waste"][..., 1:]
        if max_waste.dim() > 1 and max_waste.size(-1) == td["real_waste"].size(-1):
            max_waste = max_waste[..., 1:]
        elif max_waste.dim() == 1:
            max_waste = max_waste.unsqueeze(-1)

        overflows = (real_waste >= max_waste).float().sum(-1)

        # Store for logging
        td["real_collection"] = collection
        td["cost"] = total_cost
        td["real_overflows"] = overflows

        # Store for GDPO
        td["reward_waste"] = collection
        td["reward_cost"] = -total_cost
        td["reward_overflow"] = -overflows

        reward = self.waste_weight * collection - self.cost_weight * total_cost - self.overflow_penalty * overflows
        return reward
