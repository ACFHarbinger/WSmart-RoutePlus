"""
Regular Packing Algorithm Module.

This module implements a fixed-frequency packing algorithm. Bins are selected
if the current day matches their assigned schedule frequency (e.g., every 3 days).

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_regular import RegularSelection
    >>> strategy = RegularSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, SelectionContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.DETERMINISTIC,
)
@PackingRegistry.register("regular")
class RegularSelection(IPackingAlgorithm):
    """Periodic collection strategy.

    Attributes:
        None
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """Select all bins if it is a scheduled collection day.

        Args:
            context (SelectionContext): The selection context providing current_day and threshold (frequency).

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.
        """
        current_day = getattr(context, "current_day", 1)
        # We want to run on day 1, and then on day X+1, 2X+1, etc.
        # where X is the frequency (threshold).
        threshold = getattr(context, "threshold", 1)  # Using threshold as frequency
        if (current_day - 1) % threshold != 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "RegularSelection"})

        # Find bins where current fill >= min_fill
        eligible_bins = []
        if hasattr(context, "current_fill") and context.current_fill is not None and len(context.current_fill) > 0:
            min_fill = getattr(self, "min_fill", 0)
            for i, fill in enumerate(context.current_fill):
                if fill >= min_fill:
                    eligible_bins.append(i + 1)  # 1-based indexing for routing
        else:
            eligible_bins = (context.bin_ids + 1).tolist() if hasattr(context, "bin_ids") else []
        return eligible_bins, SearchContext.initialize(selection_metrics={"strategy": "RegularSelection"})
