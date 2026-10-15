"""
Stochastic Regret Packing Algorithm Module.

This module implements the Expected Overflow Regret strategy. It calculates
the mathematically exact expected overflow volume of a bin if its collection
is deferred to the next day, assuming normally distributed accumulation rates.

Attributes:
    StochasticRegretSelection(IPackingAlgorithm): Packing algorithm based on Expected Overflow Regret (EOR).

Example:
    >>> from logic.src.policies.packing.stochastic_regret import StochasticRegretSelection
    >>> strategy = StochasticRegretSelection()
    >>> bins, ctx = strategy.pack(context)
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
)
@PackingRegistry.register("stochastic_regret")
class StochasticRegretSelection(IPackingAlgorithm):
    """Selection strategy based on Expected Overflow Regret (EOR).

    Attributes:
        None

    Example:
        >>> from logic.src.policies.packing.stochastic_regret import StochasticRegretSelection
        >>> strategy = StochasticRegretSelection()
        >>> bins, ctx = strategy.pack(context)
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins based on the mathematical expectation of their overflow.

        Args:
            context (PackingContext): The selection context providing current_fill,
                accumulation_rates, and std_deviations.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.

        Raises:
            ValueError: If stochastic parameters are missing in context.
        """
        if context.accumulation_rates is None or context.std_deviations is None:
            raise ValueError("StochasticRegretSelection requires both accumulation_rates and std_deviations.")

        # Remaining capacity
        c = context.max_fill - context.current_fill
        mu = context.accumulation_rates
        sigma = context.std_deviations

        # Prevent division by zero for deterministic bins
        safe_sigma = np.where(sigma == 0, 1e-9, sigma)

        # Z-score for the remaining capacity
        Z = (c - mu) / safe_sigma

        # Closed-form expected value of max(0, X - c) for X ~ N(mu, sigma^2)
        expected_overflow = sigma * norm.pdf(Z) + (mu - c) * (1 - norm.cdf(Z))

        # Fallback for purely deterministic edge cases
        expected_overflow = np.where(sigma == 0, np.maximum(0, mu - c), expected_overflow)

        # Select bins where the expected overflow exceeds the acceptable threshold
        # (context.threshold here acts as the parameter \gamma from the paper)
        mandatory_indices = np.nonzero(expected_overflow > context.threshold)[0]

        # Return 1-based indexing for the routing engine
        return (mandatory_indices + 1).tolist(), SearchContext.initialize(
            selection_metrics={"strategy": "StochasticRegretSelection"}
        )
