"""
Lookahead Selection Strategy Module.

This module implements a predictive strategy that selects bins based on
future projections. It simulates future days to identify critical bins that
will overflow soon or are efficient to collect now.

Attributes:
    None

Example:
    >>> from logic.src.policies.mandatory.selection_lookahead import LookaheadSelection
    >>> strategy = LookaheadSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.constants import MAX_CAPACITY_PERCENT
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, SelectionContext
from logic.src.interfaces.mandatory_selection import IMandatorySelectionStrategy
from logic.src.policies.mandatory_selection.base.selection_registry import MandatorySelectionRegistry


@GlobalRegistry.register(
    PolicyTag.SELECTION,
    PolicyTag.HEURISTIC,
    PolicyTag.MULTI_PERIOD,
)
@MandatorySelectionRegistry.register("lookahead")
class LookaheadSelection(IMandatorySelectionStrategy):
    """Predictive selection strategy looking ahead to synchronize collections.

    Attributes:
        None
    """

    def _should_bin_be_collected(self, current_fill_level: float, accumulation_rate: float) -> bool:
        """Check if bin overflows today.

        Args:
            current_fill_level (float): Current fill level.
            accumulation_rate (float): Accumulation rate.

        Returns:
            bool: True if bin overflows today.
        """
        return current_fill_level + accumulation_rate >= MAX_CAPACITY_PERCENT

    def _update_fill_levels_after_first_collection(
        self, bin_indices: List[int], mandatory_bins: List[int], current_fill_levels: np.ndarray
    ) -> np.ndarray:
        """
        Simulate collection for mandatory_bins by setting fill levels to 0.

        Args:
            bin_indices: List of bin indices.
            mandatory_bins: List of mandatory bins.
            current_fill_levels: Array of current fill levels.

        Returns:
            np.ndarray: Updated fill levels after collection.
        """
        for i in bin_indices:
            if i in mandatory_bins:
                current_fill_levels[i] = 0
        return current_fill_levels

    def _initialize_lists_bins(self, n_bins: int) -> List[int]:
        """Initialize list for next collection days.

        Args:
            n_bins (int): Number of bins.

        Returns:
            List[int]: List of zeros.
        """
        return [0] * n_bins

    def _calculate_next_collection_days(
        self,
        bin_indices: List[int],
        mandatory_bins: List[int],
        current_fill_levels: np.ndarray,
        accumulation_rates: np.ndarray,
        current_collection_day: int = 0,
    ) -> List[int]:
        """
        Calculate when collected bins would overflow again.

        Args:
            bin_indices: List of bin indices.
            mandatory_bins: List of mandatory bins.
            current_fill_levels: Array of current fill levels.
            accumulation_rates: Array of accumulation rates.
            current_collection_day: Today's day.

        Returns:
            List[int]: List of next collection days.
        """
        next_collection_days = self._initialize_lists_bins(len(bin_indices))
        # Work on a copy to avoid side effects
        temporary_fill_levels = current_fill_levels.copy()
        for i in mandatory_bins:
            current_day = current_collection_day
            rate = accumulation_rates[i]
            if rate <= 0:
                continue

            while temporary_fill_levels[i] < MAX_CAPACITY_PERCENT:
                temporary_fill_levels[i] = temporary_fill_levels[i] + rate
                current_day = current_day + 1

            next_collection_days[i] = current_day  # assuming collection happens at the beginning of the day
        return next_collection_days

    def _get_next_collection_day(
        self,
        bin_indices: List[int],
        mandatory_bins: List[int],
        current_fill_levels: np.ndarray,
        accumulation_rates: np.ndarray,
        current_collection_day: int = 0,
    ) -> int:
        """
        Find the earliest overflow day among the currently selected bins.

        Args:
            bin_indices: List of bin indices.
            mandatory_bins: List of mandatory bins.
            current_fill_levels: Array of current fill levels.
            accumulation_rates: Array of accumulation rates.
            current_collection_day: Today's day.

        Returns:
            int: The earliest overflow day.
        """
        next_collection_days = self._calculate_next_collection_days(
            bin_indices,
            mandatory_bins,
            current_fill_levels,
            accumulation_rates,
            current_collection_day=current_collection_day,
        )
        next_collection_days_array = np.array(next_collection_days)
        # Find minimum non-zero day
        non_zero_indices = np.nonzero(next_collection_days_array)
        if len(non_zero_indices[0]) == 0:
            return 0

        next_collection_day = np.min(next_collection_days_array[non_zero_indices])
        return int(next_collection_day)

    def _add_bins_to_collect(
        self,
        bin_indices: List[int],
        next_collection_day: int,
        mandatory_bins: List[int],
        current_fill_levels: np.ndarray,
        accumulation_rates: np.ndarray,
        current_collection_day: int = 0,
    ) -> List[int]:
        """
        Add bins that would overflow before the next collection day.

        Args:
            bin_indices: List of bin indices.
            next_collection_day: The next collection day.
            mandatory_bins: List of mandatory bins.
            current_fill_levels: Array of current fill levels.
            accumulation_rates: Array of accumulation rates.
            current_collection_day: Today's day in the collection cycle.

        Returns:
            List[int]: List of bins to collect.
        """
        for i in bin_indices:
            if i in mandatory_bins:
                continue
            else:
                for j in range(current_collection_day + 1, next_collection_day):
                    # Accumulation counts the days from today, matching the vectorized selector.
                    if (
                        current_fill_levels[i] + (j - current_collection_day) * accumulation_rates[i]
                        >= MAX_CAPACITY_PERCENT
                    ):
                        mandatory_bins.append(i)
                        break
        return mandatory_bins

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """Select bins based on lookahead logic.

        Identifies bins that require immediate collection and predicts future
        overflows to bundle collections optimally.

        Args:
            context (SelectionContext): The selection context.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.
        """
        try:
            from logic.src.pipeline.simulations.actions.base import _record_live_params

            _record_live_params(
                "consumer_lookahead",
                {"current_collection_day": int(getattr(context, "current_collection_day", 0))},
            )
        except Exception:
            pass
        if context.accumulation_rates is None:
            return [], SearchContext.initialize(selection_metrics={"strategy": "LookaheadSelection"})

        current_fill_levels = context.current_fill
        accumulation_rates = context.accumulation_rates
        n_bins = len(current_fill_levels)
        bin_indices = list(range(n_bins))

        mandatory_bins = []
        for i in bin_indices:
            if self._should_bin_be_collected(current_fill_levels[i], accumulation_rates[i]):
                mandatory_bins.append(i)

        if mandatory_bins:
            # Create a copy for simulation
            simulated_fill_levels = self._update_fill_levels_after_first_collection(
                bin_indices, mandatory_bins, current_fill_levels.copy()
            )

            next_collection_day = self._get_next_collection_day(
                bin_indices,
                mandatory_bins,
                simulated_fill_levels,
                accumulation_rates,
                current_collection_day=context.current_collection_day,
            )

            if next_collection_day > context.current_collection_day:
                mandatory_bins = self._add_bins_to_collect(
                    bin_indices,
                    next_collection_day,
                    mandatory_bins,
                    current_fill_levels,  # Original fill levels
                    accumulation_rates,
                    current_collection_day=context.current_collection_day,
                )

        # Convert back to 1-based IDs
        return [i + 1 for i in mandatory_bins], SearchContext.initialize(
            selection_metrics={"strategy": "LookaheadSelection"}
        )
