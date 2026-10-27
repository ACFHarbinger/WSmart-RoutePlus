"""Regular packing: collect every eligible bin on scheduled days.

The period is ``context.threshold`` days. Scheduled days are
``1, 1+period, 1+2*period, …`` (``spec.regular``).

Attributes:
    RegularSelection: Periodic collection packing algorithm.
"""

from typing import Any, List, Tuple

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry
from logic.src.tracking.execution_trace import solution_record
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

    def pack(self, context: PackingContext, **kwargs: Any) -> Tuple[List[int], SearchContext]:
        """Select eligible bins if today is a scheduled collection day.

        Args:
            context: Day, period (``threshold``) and fills.
            kwargs: Extra context, including optional trace and parent_id.

        Returns:
            Tuple[List[int], SearchContext]: 1-based IDs and a fresh search context.
        """
        current_day = getattr(context, "current_day", 1)
        frequency = getattr(context, "threshold", 1)
        min_fill = getattr(self, "min_fill", 0.0)
        fills = context.current_fill
        if fills is None or len(fills) == 0:
            selected = []
        else:
            mask = regular_rule(fills, int(current_day), float(frequency), float(min_fill))
            selected = selected_ids(mask)
        trace = kwargs.get("trace") or getattr(context, "trace", None)
        if trace is not None:
            parent_id = kwargs.get("parent_id") or getattr(context, "trace_parent_id", None)
            if parent_id is None and trace.events:
                parent_id = trace.events[-1]["id"]
            before_routes = kwargs.get("routes") or getattr(context, "routes", None) or []
            after_routes = [selected] if selected else []
            trace.add(
                name="regular",
                kind="operator",
                parent_id=parent_id,
                changed_solution=bool(after_routes != before_routes),
                arguments={
                    "frequency": float(frequency),
                    "current_day": int(current_day),
                    "selected_bins": list(selected),
                },
                before=solution_record(before_routes),
                after=solution_record(after_routes),
            )
        return selected, SearchContext.initialize(selection_metrics={"strategy": "RegularSelection"})
