"""Last-minute packing: collect bins at or above a fill threshold.

1.0.0 last-minute is the fill-threshold rule in ``spec.last_minute``
(RQ-cursor-3). EOQ last-minute is the unreleased ``eoq_last_minute`` key.

Attributes:
    LastMinuteSelection: Threshold-based reactive packing algorithm.
"""

from typing import List, Tuple

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm

from .base import PackingRegistry
from .spec import last_minute as last_minute_rule
from .spec import selected_ids


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
)
@PackingRegistry.register("last_minute")
class LastMinuteSelection(IPackingAlgorithm):
    """Collect if current fill >= threshold (both percent of capacity)."""

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins that meet the fill threshold.

        Args:
            context: Fill levels and threshold.

        Returns:
            Tuple[List[int], SearchContext]: 1-based IDs and a fresh search context.
        """
        mask = last_minute_rule(context.current_fill, context.threshold)
        return selected_ids(mask), SearchContext.initialize(selection_metrics={"strategy": "LastMinuteSelection"})
