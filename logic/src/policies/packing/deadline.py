"""Deadline-Driven Packing Algorithm Module.

This strategy computes the exact floor of days remaining until guaranteed
overflow based on the expected accumulation rate. It obligates collection
for any bin whose deadline is less than or equal to a defined lookahead horizon.

Attributes:
    None

Example:
    >>> from logic.src.policies.packing.deadline import DeadlineDrivenSelection
    >>> strategy = DeadlineDrivenSelection()
    >>> bins = strategy.pack(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.enums import PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry
from logic.src.utils.registry import ALL


@ALL.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.DETERMINISTIC,
)
@PackingRegistry.register("deadline")
class DeadlineDrivenSelection(IPackingAlgorithm):
    """Temporal packing algorithm based on exact day-to-overflow calculation.

    Attributes:
        None
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Selects bins that will reach maximum capacity within the lookahead horizon.

        Args:
            context: Selection context containing current fill levels,
                                        accumulation rates, and horizon thresholds.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs and search context.
        """
        if context.accumulation_rates is None:
            raise ValueError("DeadlineDrivenSelection requires accumulation_rates.")

        # The threshold is the deadline horizon in days
        horizon_days = context.threshold

        rem_capacity = context.max_fill - context.current_fill

        # Prevent division by zero for bins with zero expected accumulation
        mu = np.where(context.accumulation_rates == 0, 1e-9, context.accumulation_rates)

        # Expected days until the bin hits capacity. Flooring that count
        # collects bins that are still under capacity at the end of the horizon
        # (floor(2.9) <= 2). Compare the unfloored time instead.
        days_to_overflow = rem_capacity / mu

        # Select bins that reach capacity within the requested horizon
        mandatory_indices = np.nonzero(days_to_overflow <= horizon_days)[0]

        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "DeadlineDrivenSelection"}
        )
