"""
CVRPP Environment implementation.

Attributes:
    CVRPPEnv: CVRPP environment.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("cvrpp", num_loc=50)
    >>> td = env.reset()
"""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.routing.vrpp import VRPPEnv


class CVRPPEnv(VRPPEnv):
    """
    Capacitated VRPP: VRPP with vehicle capacity constraints.

    Attributes:
        name: Name of the environment.
    """

    name: str = "cvrpp"

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        """Initialize CVRPP state with capacity tracking.

        Args:
            tensordict: Input TensorDict containing graph structure and node properties.

        Returns:
            TensorDict: Initialized CVRPP state with capacity tracking.
        """
        tensordict = super()._reset_instance(tensordict)

        bs = tensordict.batch_size[0]
        device = tensordict.device

        # Track remaining capacity
        capacity = tensordict.get("capacity", torch.ones(bs, device=device) * 100)
        tensordict["capacity"] = capacity  # Ensure it's in the TensorDict for _step
        tensordict["remaining_capacity"] = capacity.clone()
        tensordict["collected_waste"] = torch.zeros(bs, device=device)
        tensordict["collected"] = tensordict["collected_waste"]  # Alias

        return tensordict

    def _reset(self, tensordict: Optional[TensorDict] = None, **kwargs) -> TensorDict:  # type: ignore[override]
        """Reset the environment.

        Args:
            tensordict: Input TensorDict containing graph structure and node properties.
            kwargs: Additional keyword arguments.

        Returns:
            TensorDict: Reset environment state.
        """
        return super()._reset(tensordict, **kwargs)

    def _step_instance(self, tensordict: TensorDict) -> TensorDict:
        """reset.

        Args:
            tensordict (Optional[TensorDict]): Description of tensordict.
            kwargs (Any): Additional keyword arguments.

        Returns:
            Any: Updated TensorDict or tensor containing the result.
        """
        """Execute action with capacity tracking."""
        action = tensordict["action"]
        if action.dim() > 1:
            action = action.squeeze(-1)

        # Snapshot the node/at-depot status before super() advances current_node.
        waste = tensordict["waste"]
        waste_at_node = waste.gather(1, action.unsqueeze(-1)).squeeze(-1) if waste.dim() > 1 else waste
        at_depot = action == 0

        # Delegate distance / visited / current_node / tour updates, and
        # VRPP's own cumulative collected_waste, to VRPPEnv/OpsMixin. Without
        # this call, current_node/visited/tour_length/tour never advance past
        # their reset values -- the bug this fixes: CVRPPEnv previously
        # replaced the base state transition entirely instead of layering
        # capacity tracking on top of it.
        tensordict = super()._step_instance(tensordict)

        # CVRPP-specific: track *per-trip* remaining capacity, reset at each
        # depot return, separate from VRPP's all-time collected_waste total.
        tensordict["remaining_capacity"] = torch.where(
            at_depot,
            tensordict["capacity"],
            tensordict["remaining_capacity"] - waste_at_node,
        )
        tensordict["collected"] = tensordict["collected_waste"]  # Alias

        return tensordict

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Step the environment.

        Args:
            tensordict: Input TensorDict containing action.

        Returns:
            TensorDict: Environment state after action.
        """
        return OpsMixin._step(self, tensordict)

    def _get_action_mask(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Mask nodes that would exceed capacity, respecting mandatory constraints.

        Applies capacity constraints on top of base VRPP mask, then
        re-applies mandatory logic to determine depot validity.

        Args:
            tensordict: Input TensorDict containing graph structure and node properties.

        Returns:
            torch.Tensor: Action mask.
        """
        # Get base mask (without mandatory depot logic applied yet)
        base_mask = ~tensordict["visited"].clone()

        # Mask nodes whose waste exceeds remaining capacity
        waste = tensordict["waste"]
        remaining = tensordict["remaining_capacity"].unsqueeze(-1)
        exceeds_capacity = waste > remaining

        mask = base_mask & ~exceeds_capacity

        # Mandatory routing logic (same as parent but considering capacity)
        mandatory = tensordict.get("mandatory", None)
        if mandatory is not None:
            # Handle dimension mismatch (if mandatory excludes depot)
            if mandatory.size(-1) == mask.size(-1) - 1:
                mandatory = torch.cat([torch.zeros_like(mandatory[:, :1], dtype=torch.bool), mandatory], dim=1)

            # Pending mandatory bins: mandatory AND not yet visited AND within capacity
            pending_mandatory = mandatory & mask

            # Check if any mandatory bins remain (excluding depot at index 0)
            has_pending_mandatory = pending_mandatory[:, 1:].any(dim=1)

            # Depot is valid only if no pending mandatory bins remain
            mask[:, 0] = ~has_pending_mandatory
        else:
            # No mandatory constraint: depot always valid
            mask[:, 0] = True

        return mask
