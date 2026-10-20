"""Regular packing: collect every eligible bin on scheduled days.

The period is ``context.threshold`` days. Scheduled days are
``1, 1+period, 1+2*period, …`` (``spec.regular``).

Attributes:
    RegularSelection: Periodic collection packing algorithm.
"""

from typing import List, Tuple

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry
from logic.src.utils.registry import ALL

from .spec import regular as regular_rule
from .spec import selected_ids


@ALL.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.DETERMINISTIC,
)
@PackingRegistry.register("regular")
class RegularSelection(IPackingAlgorithm):
    """Periodic collection on days matching the configured period."""

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select eligible bins if today is a scheduled collection day.

        Args:
            context: Day, period (``threshold``) and fills.

        Returns:
            Tuple[List[int], SearchContext]: 1-based IDs and a fresh search context.
        """
        current_day = getattr(context, "current_day", 1)
        frequency = getattr(context, "threshold", 1)
        min_fill = getattr(self, "min_fill", 0.0)
        fills = context.current_fill
        if fills is None or len(fills) == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "RegularSelection"})
        mask = regular_rule(fills, int(current_day), float(frequency), float(min_fill))
        return selected_ids(mask), SearchContext.initialize(selection_metrics={"strategy": "RegularSelection"})
