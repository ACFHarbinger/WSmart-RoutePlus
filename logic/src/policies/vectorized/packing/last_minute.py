"""Last-minute packing adapter (batched Torch).

Implements ``packing.spec.last_minute`` on ``[B, N]`` fraction tensors.

Attributes:
    LastMinuteSelector: Threshold-based reactive selection policy.
"""

from typing import Any, Optional

import torch

from .base import VectorizedSelector
from .spec import last_minute as last_minute_rule


class LastMinuteSelector(VectorizedSelector):
    """Threshold-based reactive packing algorithm."""

    def __init__(self, threshold: float = 70.0) -> None:
        """Initialize the last-minute selector.

        Args:
            threshold: Fill level threshold in percent (0-100), matching
                ``pk_last_minute.yaml`` (cf70 -> 70). Fills passed to
                ``select`` are fractions in ``[0, 1]``.
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
            fill_levels: Current fill levels of bins ``[B, N]``.
            threshold: Override for the default fill threshold (percent).
            kwargs: Unused extra features.

        Returns:
            torch.Tensor: Boolean mask ``[B, N]``; depot is False.
        """
        thresh = threshold if threshold is not None else self.threshold
        return last_minute_rule(fill_levels, float(thresh))
