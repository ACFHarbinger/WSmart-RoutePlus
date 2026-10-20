"""Last-minute packing: collect bins at or above a fill threshold.

1.0.0 last-minute is the fill-threshold rule in ``spec.last_minute``
(RQ-cursor-3). EOQ last-minute is the unreleased ``eoq_last_minute`` key.

Attributes:
    LastMinuteSelection: Threshold-based reactive packing algorithm.
"""

import sys
from contextlib import suppress
from typing import List, Tuple

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.utils.registry import ALL

from .base import PackingRegistry
from .spec import last_minute as last_minute_rule
from .spec import selected_ids


@ALL.register(
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
        # Record the threshold that reached the consumer (#61). Looked up, not imported, so the
        # released packing path stays torch-free; a no-op outside the simulator's capture.
        recorder = sys.modules.get("logic.src.pipeline.simulations.actions.base")
        if recorder is not None:
            with suppress(Exception):
                recorder._record_live_params("consumer_last_minute", {"threshold": float(context.threshold)})
        mask = last_minute_rule(context.current_fill, context.threshold)
        return selected_ids(mask), SearchContext.initialize(selection_metrics={"strategy": "LastMinuteSelection"})
