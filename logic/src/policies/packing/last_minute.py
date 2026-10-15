"""
Last Minute Packing Algorithm Module.

This module implements the "Last Minute" or "Reactive" strategy, which
selects bins only when they exceed a certain fill threshold (e.g., 100%).

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_last_minute import LastMinuteSelection
    >>> strategy = LastMinuteSelection()
    >>> bins = strategy.pack(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, PackingContext
from logic.src.interfaces.packing import IPackingAlgorithm

from .base import PackingRegistry
from .base.eoq import resolve_trigger_threshold


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
)
@PackingRegistry.register("last_minute")
class LastMinuteSelection(IPackingAlgorithm):
    """Simple threshold-based reactive strategy.

    Logic: Collect if current_fill >= threshold (both in percent of capacity).

    Attributes:
        None
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins that exceed the fill threshold.

        Args:
            context: PackingContext with fill levels and threshold.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.
        """
        try:
            from logic.src.pipeline.simulations.actions.base import _record_live_params

            _record_live_params(
                "consumer_last_minute",
                {
                    "threshold": float(context.threshold),
                    "use_eoq_threshold": bool(getattr(context, "use_eoq_threshold", False)),
                },
            )
        except Exception:
            pass
        mandatory_mask = resolve_trigger_threshold(context)
        mandatory_indices = np.nonzero(mandatory_mask)[0]
        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "LastMinuteSelection"}
        )
