"""
Staggered Regular Selection Module.

Extends the periodic RegularSelection by assigning each bin a fixed phase
offset derived from its position in the bin list:

    φ_i = i % X      (i is the 0-based bin index, X is the period)

Bin i is mandated on operational day t if and only if:

    (t - 1) % X == φ_i

This distributes collection load uniformly across the scheduling horizon:
on every day, approximately 1/X of all bins are mandated, eliminating the
all-or-nothing load spikes of the base RegularSelection (which mandates
every bin simultaneously every X days).

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_staggered_regular import StaggeredRegularSelection
    >>> strategy = StaggeredRegularSelection()
    >>> bins, ctx = strategy.pack(context)
"""

from typing import List, Optional, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, PackingContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.DETERMINISTIC,
    PolicyTag.MULTI_PERIOD,
)
@PackingRegistry.register("staggered_regular")
class StaggeredRegularSelection(IPackingAlgorithm):
    """Phase-staggered periodic collection strategy.

    Each bin i is assigned a fixed phase offset.

    Attributes:
        min_fill (float): Minimum fill ratio for a bin to be eligible.
                          Defaults to 0.0.
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Return bin IDs whose phase offset matches today's slot in the period.

        Args:
            context (PackingContext): The selection context providing current_day,
                current_fill, and threshold (frequency).

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.
        """
        current_day: int = getattr(context, "current_day", 1)
        threshold: int = max(1, int(getattr(context, "threshold", 1)))
        min_fill: float = float(getattr(self, "min_fill", 0.0))

        current_fill: Optional[np.ndarray] = getattr(context, "current_fill", None)
        bin_ids: Optional[np.ndarray] = getattr(context, "bin_ids", None)

        # The slot within [0, X) that today maps to. Bin i is due iff φ_i == day_slot.
        day_slot: int = (current_day - 1) % threshold

        eligible_bins: List[int] = []

        if current_fill is not None and len(current_fill) > 0:
            for i, fill in enumerate(current_fill):
                # Phase offset of bin i is i % X; collect iff it matches today's slot.
                if i % threshold != day_slot:
                    continue
                if fill < min_fill:
                    continue
                eligible_bins.append(i + 1)  # 1-based indexing for routing
        elif bin_ids is not None and len(bin_ids) > 0:
            # Fall back to position-in-array index when fill data is absent.
            for idx, raw_id in enumerate(bin_ids):
                if idx % threshold == day_slot:
                    eligible_bins.append(int(raw_id) + 1)

        metrics: dict = {
            "strategy": "StaggeredRegularSelection",
            "day_slot": day_slot,
            "threshold": threshold,
            "n_selected": len(eligible_bins),
        }
        return eligible_bins, SearchContext.initialize(selection_metrics=metrics)
