"""Service level (Statistical) selection strategy.

This module provides a statistical selection strategy that predicts overflow
probabilities using mean accumulation rates and standard deviations, marking
bins for collection based on a specified confidence level.

Attributes:
    ServiceLevelSelector: Statistical overflow prediction policy.

Example:
    >>> selector = ServiceLevelSelector(confidence_factor=1.5)
    >>> mask = selector.select(fill_levels, rates, stds)
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from logic.src.constants import MAX_WASTE

from .base import VectorizedSelector


class ServiceLevelSelector(VectorizedSelector):
    """Statistical overflow prediction strategy.

    Uses mean accumulation rates and their standard deviations to predict the
    likelihood of overflow. A bin is selected if its predicted peak fill level
    (mean + confidence * std) exceeds the overflow threshold.

    Attributes:
        confidence_factor: Safety factor (z-score) for overflow risk.
    """

    def __init__(self, confidence_factor: float = 1.0, horizon_days: int = 1, **kwargs: Any) -> None:
        """Initialize the service level selector.

        Args:
            confidence_factor: Number of standard deviations for prediction.
                Higher values lead to more conservative (earlier) collections.
            horizon_days: Number of days to project into the future.
            kwargs: Additional keyword arguments.
        """
        self.confidence_factor = confidence_factor
        self.horizon_days = horizon_days

    def select(
        self,
        fill_levels: torch.Tensor,
        accumulation_rates: Optional[torch.Tensor] = None,
        std_deviations: Optional[torch.Tensor] = None,
        confidence_factor: Optional[float] = None,
        horizon_days: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Select bins statistically likely to overflow.

        Prediction Logic:
            predicted_fill = current + (horizon * rate) + (horizon * confidence * std) >= MAX_WASTE

        Args:
            fill_levels: Current fill levels [B, N].
            accumulation_rates: Mean daily waste generation per node [B, N].
            std_deviations: Standard deviation of daily generation [B, N].
            confidence_factor: Override for risk-aversion factor (threshold).
            horizon_days: Override for lookahead horizon.
            kwargs: Additional keyword arguments (e.g., 'threshold').

        Returns:
            torch.Tensor: Boolean mask [B, N] where True indicates collection.
        """
        # Support both confidence_factor and threshold (from simulation context)
        conf = confidence_factor if confidence_factor is not None else kwargs.get("threshold", self.confidence_factor)
        horizon = horizon_days if horizon_days is not None else self.horizon_days
        overflow_thresh = MAX_WASTE

        if accumulation_rates is None or std_deviations is None:
            # Without statistics, no prediction possible - select nothing
            mandatory = torch.zeros_like(fill_levels, dtype=torch.bool)
        else:
            # Statistical prediction: current + (horizon * mean) + (horizon * confidence * std)
            predicted_fill = fill_levels + (horizon * accumulation_rates) + (horizon * conf * std_deviations)
            mandatory = predicted_fill >= overflow_thresh

        # Depot is never mandatory
        mandatory = mandatory.clone()
        mandatory[:, 0] = False

        return mandatory
