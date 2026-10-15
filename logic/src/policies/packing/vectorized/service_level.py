"""Service-level packing adapter (batched Torch).

Implements ``packing.spec.service_level`` on ``[B, N]`` fraction tensors.

Attributes:
    ServiceLevelSelector: Statistical overflow prediction policy.
"""

from typing import Any, Optional

import torch

from .base import VectorizedSelector
from .spec import service_level as service_level_rule


class ServiceLevelSelector(VectorizedSelector):
    """Statistical overflow prediction strategy."""

    def __init__(self, confidence_factor: float = 1.0, horizon_days: int = 1, **kwargs: Any) -> None:
        """Initialize the service level selector.

        Args:
            confidence_factor: Number of standard deviations for prediction.
            horizon_days: Number of days to project into the future.
            kwargs: Unused extra constructor arguments.
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

        Args:
            fill_levels: Current fill levels ``[B, N]``.
            accumulation_rates: Mean daily waste generation ``[B, N]``.
            std_deviations: Standard deviation of daily generation ``[B, N]``.
            confidence_factor: Override for the z-score.
            horizon_days: Override for the lookahead horizon.
            kwargs: Accepts ``threshold`` as an alias for the z-score.

        Returns:
            torch.Tensor: Boolean mask ``[B, N]``; depot is False.
        """
        conf = confidence_factor if confidence_factor is not None else kwargs.get("threshold", self.confidence_factor)
        horizon = horizon_days if horizon_days is not None else self.horizon_days
        return service_level_rule(fill_levels, accumulation_rates, std_deviations, float(conf), int(horizon))
