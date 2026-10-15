"""
Multi-Day Overflow Probability Selection Module.

This strategy evaluates the tail probability of a bin overflowing over a
stochastic horizon of K days. Assuming daily waste generation is i.i.d Gaussian,
the total accumulation variance scales by the square root of time (sqrt(D)),
providing a much more mathematically rigorous risk assessment than linear bounds.

Attributes:
    MultiDayOverflowSelection: The Multi Day Overflow Probability Packing algorithm.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_multi_day_prob import MultiDayOverflowSelection
    >>> strategy = MultiDayOverflowSelection()
    >>> bins = strategy.pack(context)
"""

from typing import List, Tuple

import numpy as np
from scipy.stats import norm

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, PackingContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.STOCHASTIC,
    PolicyTag.MULTI_PERIOD,
)
@PackingRegistry.register("multi_day_prob")
class MultiDayOverflowSelection(IPackingAlgorithm):
    """Stochastic packing algorithm based on cumulative probability over time.

    Attributes:
        None
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Selects bins based on overflow probability within K days.

        Computes the tail probability of a bin overflowing over a stochastic
        horizon of K days. Assumes daily waste generation is i.i.d Gaussian.

        Args:
            context: The selection context.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.

        Raises:
            ValueError: If stochastic parameters (accumulation_rates or std_deviations)
                are missing in the context.
        """
        if context.accumulation_rates is None or context.std_deviations is None:
            raise ValueError("Missing stochastic parameters in context.")

        # Extract horizon parameter dynamically
        horizon_days = getattr(context, "horizon_days", 3)

        rem_capacity = context.max_fill - context.current_fill

        # Scale parameters for K days. Variance scales linearly with time,
        # meaning standard deviation scales with the square root of time.
        mu_k = context.accumulation_rates * horizon_days
        sigma_k = context.std_deviations * np.sqrt(horizon_days)

        # Prevent division by zero for deterministic cases
        safe_sigma = np.where(sigma_k == 0, 1e-9, sigma_k)

        # Compute the probability of the accumulation exceeding remaining capacity
        # P(Accumulation >= Remaining Capacity)
        Z = (rem_capacity - mu_k) / safe_sigma
        prob_overflow = 1.0 - norm.cdf(Z)

        # Deterministic fallback for bins with 0 variance
        prob_overflow = np.where(sigma_k == 0, (mu_k >= rem_capacity).astype(float), prob_overflow)

        # Here context.threshold acts as the maximum acceptable risk probability (e.g., 0.85)
        mandatory_indices = np.nonzero(prob_overflow >= context.threshold)[0]

        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "MultiDayOverflowSelection"}
        )
