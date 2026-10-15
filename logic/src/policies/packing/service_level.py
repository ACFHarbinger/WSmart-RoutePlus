"""Service-level packing: collect bins that project to overflow.

Uses the linear confidence bound in ``spec.service_level``.

Attributes:
    ServiceLevelSelection: Statistical overflow-projection packing algorithm.
"""

from typing import List, Tuple

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry

from .spec import selected_ids
from .spec import service_level as service_level_rule


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.STOCHASTIC,
)
@PackingRegistry.register("service_level")
class ServiceLevelSelection(IPackingAlgorithm):
    """Collect bins whose worst-case linear projection reaches capacity."""

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins statistically likely to overflow within the horizon.

        Args:
            context (PackingContext): Fills, rates, stds, z (``threshold``), horizon.

        Returns:
            Tuple[List[int], SearchContext]: 1-based IDs and a fresh search context.
        """
        mask = service_level_rule(
            context.current_fill,
            context.accumulation_rates,
            context.std_deviations,
            float(context.threshold),
            int(getattr(context, "horizon_days", 1)),
        )
        return selected_ids(mask), SearchContext.initialize(selection_metrics={"strategy": "ServiceLevelSelection"})
