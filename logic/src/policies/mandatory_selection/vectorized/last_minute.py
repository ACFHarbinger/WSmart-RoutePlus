"""Last-minute selection strategy.

This module provides a threshold-based reactive selection strategy that marks
bins for collection only when their fill levels exceed a specified limit.

Attributes:
    LastMinuteSelector: Threshold-based reactive selection policy.

Example:
    >>> selector = LastMinuteSelector(threshold=80)
    >>> mask = selector.select(fill_levels)
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import VectorizedSelector


class LastMinuteSelector(VectorizedSelector):
    """Threshold-based reactive selection strategy.

    Selects bins where the current fill level exceeds a predefined threshold.
    This strategy is simple but reactive, only triggering collection when bins
    are nearly full.

    Attributes:
        threshold: Default fill level threshold for selection.
    """

    def __init__(self, threshold: float = 70.0) -> None:
        """Initialize the last-minute selector.

        Args:
            threshold: Fill level threshold in percent (0-100), the same unit as the
                simulator's ms_last_minute.yaml (cf70 -> 70). Fill levels passed to
                ``select`` are fractions in [0, 1].
        """
        self.threshold = threshold

    def select(
        self,
        fill_levels: torch.Tensor,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins exceeding the fill threshold.

        Args:
            fill_levels: Current fill levels of bins [B, N].
            threshold: Override for the default fill threshold.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
        """
        thresh = threshold if threshold is not None else self.threshold
        mandatory = fill_levels >= float(thresh) / 100.0  # percent threshold vs fraction fills

        # Depot (index 0) is never mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
