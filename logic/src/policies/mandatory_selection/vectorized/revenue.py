"""Revenue selection strategy.

This module provides a revenue-based selection strategy that marks bins for
collection only when the expected profit from their waste exceeds a specified
monetary threshold.

Attributes:
    RevenueSelector: Revenue-based selection policy.

Example:
    >>> selector = RevenueSelector(threshold=10.0)
    >>> mask = selector.select(fill_levels)
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import VectorizedSelector


class RevenueSelector(VectorizedSelector):
    """Revenue-based selection strategy.

    Selects bins where the expected collection revenue exceeds a threshold.
    Revenue is calculated as the product of fill level, bin capacity, and
    revenue per kilogram.

    Attributes:
        revenue_kg: Profit per unit of mass.
        bin_capacity: Max mass per bin.
        threshold: Minimum profit to trigger collection.
    """

    def __init__(
        self,
        revenue_kg: float = 1.0,
        bin_capacity: float = 1.0,
        threshold: float = 0.0,
    ) -> None:
        """Initialize the revenue selector.

        Args:
            revenue_kg: Revenue per kg of collected waste.
            bin_capacity: Capacity of each bin in kg.
            threshold: Minimum revenue threshold for selection.
        """
        self.revenue_kg = revenue_kg
        self.bin_capacity = bin_capacity
        self.threshold = threshold

    def select(
        self,
        fill_levels: torch.Tensor,
        revenue_kg: Optional[float] = None,
        bin_capacity: Optional[float] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins where expected revenue exceeds threshold.

        Args:
            fill_levels: Current fill levels [B, N].
            revenue_kg: Override for profit rate.
            bin_capacity: Override for bin capacity.
            threshold: Override for selection threshold.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
        """
        rev = revenue_kg if revenue_kg is not None else self.revenue_kg
        cap = bin_capacity if bin_capacity is not None else self.bin_capacity

        # If cap is 1.0 (default), check if volume/density are provided
        if cap == 1.0:
            vol = kwargs.get("bin_volume", 1.0)
            dens = kwargs.get("bin_density", 1.0)
            if vol != 1.0 or dens != 1.0:
                cap = vol * dens

        thresh = threshold if threshold is not None else self.threshold

        # Expected revenue = fill_ratio * bin_capacity * revenue_per_kg
        fill_ratios = fill_levels  # fill_levels are already normalized [0, 1] in RL pipeline
        expected_revenue = fill_ratios * cap * rev
        mandatory = expected_revenue > thresh

        # Depot is never a mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
