"""EOQ last-minute packing (unreleased; excluded from 1.0.0).

RQ-cursor-3: 1.0.0 last-minute is fill-threshold only. This key stays in
``main`` and is left out of the release manifest. It uses the NumPy EOQ
helper in ``base/eoq.py``.

Attributes:
    LastMinuteEOQSelection: Last-minute with per-bin EOQ triggers.
"""

from typing import List, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm

from .base import PackingRegistry
from .base.eoq import compute_eoq_thresholds
from .spec import selected_ids


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
)
@PackingRegistry.register("eoq_last_minute")
class LastMinuteEOQSelection(IPackingAlgorithm):
    """Collect when fill meets the per-bin EOQ trigger (Harris 1913)."""

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins at or above their EOQ-derived fill trigger.

        Falls back to ``context.threshold`` when EOQ parameters are missing.

        Args:
            context: Fills and EOQ cost parameters.

        Returns:
            Tuple[List[int], SearchContext]: 1-based IDs and a fresh search context.
        """
        tau = compute_eoq_thresholds(context)
        mask = np.asarray(context.current_fill, dtype=np.float64) >= tau
        return selected_ids(mask), SearchContext.initialize(selection_metrics={"strategy": "LastMinuteEOQSelection"})
