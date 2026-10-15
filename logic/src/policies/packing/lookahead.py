"""Lookahead packing: seed today's overflows and bundle the next trip.

The rule is ``spec.lookahead``. Missing accumulation rates yield an empty set.

Attributes:
    LookaheadSelection: Predictive packing algorithm.
"""

from typing import List, Tuple

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base.registry import PackingRegistry

from .spec import lookahead as lookahead_rule
from .spec import selected_ids


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.MULTI_PERIOD,
)
@PackingRegistry.register("lookahead")
class LookaheadSelection(IPackingAlgorithm):
    """Collect today's overflows plus bins that overflow before the next trip."""

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins from the lookahead seed-and-bundle rule.

        Args:
            context (PackingContext): Fills, rates and current collection day.

        Returns:
            Tuple[List[int], SearchContext]: 1-based IDs and a fresh search context.
        """
        mask = lookahead_rule(
            context.current_fill,
            context.accumulation_rates,
            float(getattr(context, "current_collection_day", 0)),
        )
        return selected_ids(mask), SearchContext.initialize(selection_metrics={"strategy": "LookaheadSelection"})
