"""Regular selection strategy.

This module provides a periodic collection strategy that marks all bins for
collection on scheduled days based on a fixed frequency interval.

Attributes:
    RegularSelector: Periodic collection strategy policy.

Example:
    >>> selector = RegularSelector(frequency=3)
    >>> mask = selector.select(fill_levels, current_day=4)
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch

from .base import VectorizedSelector


class RegularSelector(VectorizedSelector):
    """Periodic collection strategy.

    Selects all bins on scheduled collection days based on a fixed frequency.
    If it is a collection day for a specific batch, all active bins in that
    batch are marked as mandatory.

    Attributes:
        frequency: Default collection interval in days.
    """

    def __init__(self, frequency: int = 3) -> None:
        """Initialize the regular selector.

        Args:
            frequency: Collection interval in days (e.g., 3 means collect every 3rd day).
        """
        self.frequency = frequency

    def select(
        self,
        fill_levels: torch.Tensor,
        current_day: Optional[Union[torch.Tensor, int]] = None,
        frequency: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select all bins if today is a scheduled collection day.

        Args:
            fill_levels: Current fill levels of bins [B, N].
            current_day: The simulation day index (scalar or [B]).
            frequency: Override for the default frequency interval.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device
        freq = frequency if frequency is not None else self.frequency

        if freq <= 0:
            # Collect every day
            mandatory = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        elif current_day is None:
            # No day info - assume collection day
            mandatory = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
        else:
            # Ensure current_day is a tensor
            if not isinstance(current_day, torch.Tensor):
                current_day = torch.tensor(current_day, device=device)

            # Expand to batch if scalar
            if current_day.dim() == 0:
                current_day = current_day.expand(batch_size)

            # Scheduled collection: collect ALL bins on days 1, 1+freq, 1+2*freq, ...
            # i.e. (current_day - 1) % freq == 0
            is_collection_day = ((current_day - 1) % freq) == 0  # (B,)

            # Broadcast to (B, N)
            mandatory = is_collection_day.unsqueeze(-1).expand(batch_size, num_nodes)

        # Depot is never a mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
