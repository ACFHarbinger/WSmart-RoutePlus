"""Regular packing adapter (batched Torch).

Implements ``packing.spec.regular`` on ``[B, N]`` fraction tensors.

Attributes:
    RegularSelector: Periodic collection strategy policy.
"""

from typing import Any, Optional, Union

import torch

from .base import VectorizedSelector
from .spec import regular as regular_rule


class RegularSelector(VectorizedSelector):
    """Periodic collection strategy."""

    def __init__(self, frequency: int = 3, min_fill: float = 0.0) -> None:
        """Initialize the regular selector.

        Args:
            frequency: Collection interval in days.
            min_fill: Minimum fill fraction on a collection day.
        """
        self.frequency = frequency
        self.min_fill = min_fill

    def select(
        self,
        fill_levels: torch.Tensor,
        current_day: Optional[Union[torch.Tensor, int]] = None,
        frequency: Optional[int] = None,
        min_fill: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select eligible bins if today is a scheduled collection day.

        Args:
            fill_levels: Current fill levels of bins ``[B, N]``.
            current_day: Simulation day index (scalar or ``[B]``).
            frequency: Override for the default period.
            min_fill: Override for the default minimum fill fraction.
            kwargs: Unused extra features.

        Returns:
            torch.Tensor: Boolean mask ``[B, N]``; depot is False.
        """
        freq = frequency if frequency is not None else self.frequency
        floor = min_fill if min_fill is not None else self.min_fill
        return regular_rule(fill_levels, current_day, float(freq), float(floor))
