"""
CPTP Environment implementation.

Attributes:
    CPTPEnv: CPTP environment.

Example:
    >>> from logic.src.envs.routing import get_env
    >>> env = get_env("cptp", num_loc=50)
    >>> td = env.reset()
"""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from logic.src.envs.base.ops import OpsMixin
from logic.src.envs.routing.ptp import PTPEnv


class CPTPEnv(PTPEnv):
    """
    Capacitated Profitable Tour Problem: PTP with vehicle capacity constraints.

    Attributes:
        name: Name of the environment.
    """

    name: str = "cptp"

    # A depot visit unloads the vehicle and the day continues. The day ends when the depot is
    # chosen while the vehicle is already at the depot ("stop"), and that end is permanent.
    # Subclasses with their own end rule (TCPTP) switch this off.
    explicit_stop: bool = True

    # Mask bins that hold no waste, unless they are mandatory. Off in training; the simulator turns
    # it on for neural routing, where an empty bin only adds distance.
    mask_empty_bins: bool = False

    def _reset_instance(self, tensordict: TensorDict) -> TensorDict:
        """Initialize CPTP state with capacity tracking.

        Args:
            tensordict: Input TensorDict containing graph structure and node properties.

        Returns:
            TensorDict: Initialized CPTP state with capacity tracking.
        """
        is_resuming = "visited" in tensordict.keys()
        tensordict = super()._reset_instance(tensordict)

        bs = tensordict.batch_size[0]
        device = tensordict.device

        # Track remaining capacity
        capacity = tensordict.get("capacity", torch.ones(bs, device=device) * 100)
        tensordict["capacity"] = capacity  # Ensure it's in the TensorDict for _step
        if not is_resuming:
            tensordict["remaining_capacity"] = capacity.clone()
            tensordict["collected_waste"] = torch.zeros(bs, device=device)
        else:
            tensordict.setdefault("remaining_capacity", capacity.clone())
            tensordict.setdefault("collected_waste", torch.zeros(bs, device=device))
        tensordict["collected"] = tensordict["collected_waste"]  # Alias

        if not is_resuming or "finished" not in tensordict.keys():
            tensordict["finished"] = torch.zeros(bs, dtype=torch.bool, device=device)

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

        # Choosing the depot from the depot ends the day (also as the very first action).
        was_at_depot = tensordict["current_node"].reshape(-1) == 0
        finished = tensordict.get("finished", torch.zeros_like(at_depot))
        if self.explicit_stop:
            finished = finished | (at_depot & was_at_depot)

        # Delegate distance / visited / current_node / tour updates, and
        # PTP's own cumulative collected_waste, to PTPEnv/OpsMixin. Without
        # this call, current_node/visited/tour_length/tour never advance past
        # their reset values -- the bug this fixes: CPTPEnv previously
        # replaced the base state transition entirely instead of layering
        # capacity tracking on top of it.
        tensordict = super()._step_instance(tensordict)

        # CPTP-specific: track *per-trip* remaining capacity, reset at each
        # depot return, separate from PTP's all-time collected_waste total.
        # Clamp so an illegal (unmasked) action cannot drive capacity negative.
        tensordict["remaining_capacity"] = torch.where(
            at_depot,
            tensordict["capacity"],
            (tensordict["remaining_capacity"] - waste_at_node).clamp(min=0),
        )
        tensordict["collected"] = tensordict["capacity"] - tensordict["remaining_capacity"]
        tensordict["finished"] = finished

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

        Applies capacity constraints on top of base PTP mask, then
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

        if self.mask_empty_bins:
            keep = waste > 0
            pending = tensordict.get("mandatory", None)
            if pending is not None:
                if pending.size(-1) == keep.size(-1) - 1:
                    pending = torch.cat([torch.zeros_like(pending[:, :1], dtype=torch.bool), pending], dim=1)
                keep = keep | pending.bool()
            keep[:, 0] = True
            mask = mask & keep

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

        # A finished day stays at the depot; the padding steps cost nothing.
        finished = tensordict.get("finished", None)
        if self.explicit_stop and finished is not None:
            mask[finished, 1:] = False
            mask[finished, 0] = True

        return mask

    def _check_done(self, tensordict: TensorDict) -> torch.Tensor:
        """
        The day is done once the vehicle has stopped at the depot, or is at the depot with
        no bin left that it could collect. A plain depot return only unloads.

        Args:
            tensordict: Input TensorDict containing the state after the step.

        Returns:
            torch.Tensor: Boolean tensor indicating if the episode is done, shape (batch,).
        """
        if not self.explicit_stop:
            return super()._check_done(tensordict)
        current = tensordict["current_node"].reshape(-1)
        step = tensordict["i"].reshape(-1)
        mask = tensordict.get("action_mask", None)
        if mask is None:
            mask = self._get_action_mask(tensordict)
        exhausted = (current == 0) & (step > 0) & ~mask[:, 1:].any(dim=1)
        return tensordict["finished"] | exhausted
