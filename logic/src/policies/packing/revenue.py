"""
Revenue Packing Algorithm Module.

This module implements a value-based packing algorithm. Bins are selected
if the estimated revenue of their current content exceeds a threshold.

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_revenue import RevenueThresholdSelection
    >>> strategy = RevenueThresholdSelection()
    >>> bins = strategy.pack(context)
"""

from typing import List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm

from .base.eoq import resolve_trigger_threshold
from .base.registry import PackingRegistry


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.PROFIT_AWARE,
)
@PackingRegistry.register("revenue")
class RevenueThresholdSelection(IPackingAlgorithm):
    """Revenue-based packing algorithm.

    Attributes:
        None
    """

    def __init__(self, revenue_kg: Optional[float] = None, revenue_threshold: Optional[float] = None) -> None:
        """
        Args:
            revenue_kg: Revenue per kilogram; ``None`` uses ``context.revenue_kg``.
            revenue_threshold: Selection threshold; ``None`` uses ``context.threshold``.
        """
        self.revenue_kg = revenue_kg
        self.revenue_threshold = revenue_threshold

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins where expected revenue exceeds the threshold.

        Args:
            context: PackingContext with revenue parameters and threshold.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.
        """
        bin_cap = context.bin_volume * context.bin_density
        fill_ratios = context.current_fill / context.max_fill
        revenue_kg = self.revenue_kg if self.revenue_kg is not None else context.revenue_kg
        expected_revenue = fill_ratios * bin_cap * revenue_kg

        if getattr(context, "use_eoq_threshold", False):
            # When EOQ is active, the trigger is the fill-level, but we still
            # check revenue-positivity as a secondary guard.
            mandatory_mask = resolve_trigger_threshold(context)
            mandatory_indices = np.nonzero(mandatory_mask & (expected_revenue > 0))[0]
        else:
            threshold = self.revenue_threshold if self.revenue_threshold is not None else context.threshold
            mandatory_indices = np.nonzero(expected_revenue > threshold)[0]

        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "RevenueThresholdSelection"}
        )
