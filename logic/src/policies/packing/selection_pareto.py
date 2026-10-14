"""
Pareto-Front Packing Algorithm Module.

This module implements a sophisticated combinatorial packing algorithm based on
multi-objective optimization. Instead of relying on a single scalar score, it
evaluates bins on a bi-objective Pareto front:
    1. Urgency: Minimizing the expected days until overflow.
    2. Routing Efficiency: Minimizing the spatial distance to the depot (as a
       proxy for insertion cost).

Any bin that lies on the non-dominated Pareto front is strictly obligated for
collection. This ensures that the policy never leaves behind a bin that is either
critically urgent or trivially cheap to collect without a justified trade-off.

Attributes:
    ParetoFrontSelection(IPackingAlgorithm): Combinatorial packing algorithm utilizing non-dominated sorting.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_pareto import ParetoFrontSelection
    >>> strategy = ParetoFrontSelection()
    >>> bins = strategy.select_bins(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import SearchContext, SelectionContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
    PolicyTag.MULTI_PERIOD,
)
@PackingRegistry.register("pareto_front")
class ParetoFrontSelection(IPackingAlgorithm):
    """Combinatorial packing algorithm utilizing non-dominated sorting.

    Attributes:
        None
    """

    def select_bins(self, context: SelectionContext) -> Tuple[List[int], SearchContext]:
        """Selects the non-dominated set of bins based on urgency and routing cost.

        Evaluates bins on a bi-objective Pareto front of urgency (days to overflow)
        and routing efficiency (distance to depot).

        Args:
            context (SelectionContext): The selection context.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.

        Raises:
            ValueError: If distance_matrix or accumulation_rates are missing.
        """
        if context.distance_matrix is None:
            raise ValueError("ParetoFrontSelection requires a distance_matrix in the context.")
        if context.accumulation_rates is None:
            raise ValueError("ParetoFrontSelection requires accumulation_rates in the context.")

        rem_capacity = context.max_fill - context.current_fill

        # Prevent division by zero for bins with zero expected accumulation
        mu = np.where(context.accumulation_rates == 0, 1e-9, context.accumulation_rates)

        # Objective 1: Urgency (Minimize days to overflow)
        days_to_overflow = rem_capacity / mu

        # Objective 2: Routing Cost Proxy (Minimize distance to depot)
        # Note: distance_matrix[0] is the depot; [1:] maps to the bins.
        dist_to_depot = context.distance_matrix[0, 1:]

        mandatory = []
        n_bins = len(days_to_overflow)

        for i in range(n_bins):
            # Pre-filter: Do not evaluate bins that are comfortably far from overflowing.
            # context.threshold acts as the maximum acceptable lookahead horizon here.
            if days_to_overflow[i] > context.threshold:
                continue

            dominated = False
            for j in range(n_bins):
                if i == j:
                    continue

                # Check for strict Pareto dominance.
                # Bin j dominates Bin i if it is better or equal in both objectives,
                # and strictly better in at least one objective.
                if (days_to_overflow[j] <= days_to_overflow[i] and dist_to_depot[j] <= dist_to_depot[i]) and (
                    days_to_overflow[j] < days_to_overflow[i] or dist_to_depot[j] < dist_to_depot[i]
                ):
                    dominated = True
                    break

            if not dominated:
                mandatory.append(i + 1)

        return mandatory, SearchContext.initialize(selection_metrics={"strategy": "ParetoFrontSelection"})
