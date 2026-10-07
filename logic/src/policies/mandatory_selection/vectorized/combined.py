"""Combined selection strategy.

This module provides a mechanism to combine multiple bin selection strategies
using logical operators (AND/OR).

Attributes:
    CombinedSelector: Strategy for merging multiple selectors with logic.

Example:
    >>> s1 = RevenueSelector(threshold=10)
    >>> s2 = ServiceLevelSelector(confidence_factor=1.5)
    >>> combo = CombinedSelector([s1, s2], logic="or")
    >>> mask = combo.select(fill_levels)
"""

from __future__ import annotations

from typing import Any, List

import torch

from .base import VectorizedSelector


class CombinedSelector(VectorizedSelector):
    """Combines multiple selection strategies with Boolean logic.

    A bin is selected if the logical combination (AND/OR) of all constituent
    selectors results in a True value.

    Attributes:
        selectors: List of constituent selection strategies.
        logic: Boolean reduction method ('and' or 'or').
    """

    def __init__(self, selectors: List[VectorizedSelector], logic: str = "or") -> None:
        """Initialize the combined selector.

        Args:
            selectors: instances of VectorizedSelector to combine.
            logic: logical operator to combine masks ('or' or 'and').

        Raises:
            ValueError: if logic is not 'or' or 'and'.
        """
        self.selectors = selectors
        self.logic = logic.lower()
        if self.logic not in ["or", "and"]:
            raise ValueError(f"Unknown logic: {self.logic}. Must be 'or' or 'and'.")

    def select(
        self,
        fill_levels: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins chosen by constituent selectors according to logic.

        Args:
            fill_levels: Current fill levels [B, N].
            kwargs: Shared arguments passed to all constituent selectors.

        Returns:
            torch.Tensor: Boolean mask [B, N] of selected bins.
        """
        batch_size, num_nodes = fill_levels.shape
        device = fill_levels.device
        if self.logic == "or":
            combined = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
            for selector in self.selectors:
                mask = selector.select(fill_levels, **kwargs)
                combined = combined | mask
        else:  # logic == "and"
            combined = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)
            for selector in self.selectors:
                mask = selector.select(fill_levels, **kwargs)
                combined = combined & mask

        return combined
