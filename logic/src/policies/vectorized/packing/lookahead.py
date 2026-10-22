"""Lookahead packing adapter (batched Torch).

Implements ``packing.spec.lookahead`` on ``[B, N]`` fraction tensors.

Attributes:
    LookaheadSelector: Vectorised predictive selection policy.
"""

from typing import Any, Optional, Union

import torch

from .base import VectorizedSelector
from .spec import lookahead as lookahead_rule


class LookaheadSelector(VectorizedSelector):
    """Vectorised predictive selection looking ahead to the next collection day."""

    def __init__(self, current_collection_day: int = 0, **kwargs: Any) -> None:
        """Initialise the lookahead selector.

        Args:
            current_collection_day: Current day in the collection cycle.
            kwargs: Unused extra constructor arguments.
        """
        self.current_collection_day = current_collection_day

    def select(
        self,
        fill_levels: torch.Tensor,
        accumulation_rates: Optional[torch.Tensor] = None,
        current_collection_day: Optional[Union[int, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins predicted to overflow within the dynamic lookahead horizon.

        Args:
            fill_levels: Current fill levels, shape ``[B, N]``.
            accumulation_rates: Mean daily waste generation, shape ``[B, N]``.
            current_collection_day: Override for the reference day.
            kwargs: Unused extra features.

        Returns:
            torch.Tensor: Boolean mask ``[B, N]``; depot is False.
        """
        day = current_collection_day if current_collection_day is not None else self.current_collection_day
        return lookahead_rule(fill_levels, accumulation_rates, day)
