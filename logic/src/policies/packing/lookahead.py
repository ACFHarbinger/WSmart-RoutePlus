"""Lookahead packing: seed today's overflows and bundle the next trip.

The rule is ``spec.lookahead``. Missing accumulation rates yield an empty set.

Attributes:
    LookaheadSelection: Predictive packing algorithm.
"""

from typing import Any, List, Tuple

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base.registry import PackingRegistry
from logic.src.tracking.execution_trace import solution_record
from logic.src.utils.registry import ALL

from .spec import lookahead as lookahead_rule
from .spec import selected_ids


@ALL.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.MULTI_PERIOD,
)
@PackingRegistry.register("lookahead")
class LookaheadSelection(IPackingAlgorithm):
    """Collect today's overflows plus bins that overflow before the next trip."""

    def pack(self, context: PackingContext, **kwargs: Any) -> Tuple[List[int], SearchContext]:
        """Select bins from the lookahead seed-and-bundle rule.

        Args:
            context: Fills, rates and current collection day.
            kwargs: Extra context, including optional trace and parent_id.

        Returns:
            Tuple[List[int], SearchContext]: 1-based IDs and a fresh search context.
        """
        current_day = float(getattr(context, "current_collection_day", 0))
        mask = lookahead_rule(
            context.current_fill,
            context.accumulation_rates,
            current_day,
        )
        selected = selected_ids(mask)
        trace = kwargs.get("trace") or getattr(context, "trace", None)
        if trace is not None:
            parent_id = kwargs.get("parent_id") or getattr(context, "trace_parent_id", None)
            if parent_id is None and trace.events:
                parent_id = trace.events[-1]["id"]
            before_routes = kwargs.get("routes") or getattr(context, "routes", None) or []
            after_routes = [selected] if selected else []
            trace.add(
                name="lookahead",
                kind="operator",
                parent_id=parent_id,
                changed_solution=bool(after_routes != before_routes),
                arguments={
                    "current_collection_day": current_day,
                    "selected_bins": list(selected),
                },
                before=solution_record(before_routes),
                after=solution_record(after_routes),
            )
        return selected, SearchContext.initialize(selection_metrics={"strategy": "LookaheadSelection"})
