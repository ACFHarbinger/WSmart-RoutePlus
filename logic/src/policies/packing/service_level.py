"""
Service Level Packing Algorithm Module.

This module implements a robust optimization baseline that selects bins
to maintain a service level guarantee. It uses a linear projection of
the expected accumulation and standard deviation over a defined horizon
to ensure bins are collected before their worst-case prediction hits capacity.

Attributes:
    ServiceLevelSelection: Packing algorithm using linear projection of future fill.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_service_level import ServiceLevelSelection
    >>> strategy = ServiceLevelSelection()
    >>> bins = strategy.pack(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.constants import MAX_CAPACITY_PERCENT
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, PackingContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.STOCHASTIC,
)
@PackingRegistry.register("service_level")
class ServiceLevelSelection(IPackingAlgorithm):
    """Statistical overflow prediction strategy using linear confidence bounds.

    Attributes:
        None
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins that are statistically likely to overflow within the horizon.

        Args:
            context (PackingContext): The selection context providing current_fill,
                accumulation_rates, and std_deviations.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.
        """
        if context.accumulation_rates is None or context.std_deviations is None:
            return [], SearchContext.initialize(selection_metrics={"strategy": "ServiceLevelSelection"})

        # Extract horizon parameter dynamically (defaults to 1 for next-day projection)
        horizon_days = getattr(context, "horizon_days", 1)

        # Linear projection of worst-case fill over the horizon:
        # w_future = w_current + (D * mu) + (D * k * sigma)
        predicted_fill = (
            context.current_fill
            + (context.accumulation_rates * horizon_days)
            + (context.threshold * context.std_deviations * horizon_days)
        )

        # Identify bins where the worst-case projection exceeds capacity
        mandatory = np.nonzero(predicted_fill >= MAX_CAPACITY_PERCENT)[0] + 1
        return mandatory.tolist(), SearchContext.initialize(selection_metrics={"strategy": "ServiceLevelSelection"})
