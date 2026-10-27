"""Service-level packing: collect bins that project to overflow.

Uses the linear confidence bound in ``spec.service_level``.

Attributes:
    ServiceLevelSelection: Statistical overflow-projection packing algorithm.
"""

from typing import Any, List, Tuple

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry
from logic.src.tracking.execution_trace import solution_record
from logic.src.utils.registry import ALL

from .spec import selected_ids
from .spec import service_level as service_level_rule


@ALL.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.STOCHASTIC,
)
@PackingRegistry.register("service_level")
class ServiceLevelSelection(IPackingAlgorithm):
    """Collect bins whose worst-case linear projection reaches capacity."""

    def pack(self, context: PackingContext, **kwargs: Any) -> Tuple[List[int], SearchContext]:
        """Select bins statistically likely to overflow within the horizon.

        Args:
            context: Fills, rates, stds, z (``threshold``), horizon.
            kwargs: Extra context, including optional trace and parent_id.

        Returns:
            Tuple[List[int], SearchContext]: 1-based IDs and a fresh search context.
        """
        horizon_days = int(getattr(context, "horizon_days", 1))
        mask = service_level_rule(
            context.current_fill,
            context.accumulation_rates,
            context.std_deviations,
            float(context.threshold),
            horizon_days,
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
                name="service_level",
                kind="operator",
                parent_id=parent_id,
                changed_solution=bool(after_routes != before_routes),
                arguments={
                    "threshold": float(context.threshold),
                    "horizon_days": horizon_days,
                    "selected_bins": list(selected),
                },
                before=solution_record(before_routes),
                after=solution_record(after_routes),
            )
        return selected, SearchContext.initialize(selection_metrics={"strategy": "ServiceLevelSelection"})
