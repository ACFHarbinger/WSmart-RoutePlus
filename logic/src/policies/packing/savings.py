"""
Savings Packing Algorithm Module.

Uses the Clarke-Wright savings heuristic logic as a filter for selecting bins.
A bin is selected if it satisfies a minimum fill requirement AND there exists
at least one other sufficiently-full bin with which it shares a positive spatial
saving (i.e., routing them together is cheaper than two separate round trips).

Attributes:
    SavingsSelection(IPackingAlgorithm): Economic packing algorithm based on spatial ROI (Return on Investment).

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_savings import SavingsSelection
    >>> strategy = SavingsSelection()
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
)
@PackingRegistry.register("savings")
class SavingsSelection(IPackingAlgorithm):
    """Selection strategy based on Clarke-Wright spatial savings.

    Attributes:
        None
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select bins that offer positive routing savings with neighbors.

        Args:
            context: The selection context providing current_fill and distance_matrix.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.

        Raises:
            ValueError: If ``distance_matrix`` is missing.
        """
        if context.distance_matrix is None:
            raise ValueError("SavingsSelection requires a distance_matrix.")

        n_bins = len(context.current_fill)
        if n_bins < 2:
            return [], SearchContext.initialize(selection_metrics={"strategy": "SavingsSelection"})

        # 1. Pre-filter by fill ratio
        fill_ratios = context.current_fill / context.max_fill
        candidate_mask = fill_ratios >= context.savings_min_fill_ratio
        candidate_indices = np.nonzero(candidate_mask)[0]

        if len(candidate_indices) < 2:
            return [], SearchContext.initialize(selection_metrics={"strategy": "SavingsSelection"})

        # 2. Compute pairwise savings s_ij = d(0, i) + d(0, j) - d(i, j)
        # distance_matrix[0] is depot. Bins are index 1:n_bins+1
        d_depot = context.distance_matrix[0, 1:]

        # indices are 0-based for bins, so index i in context maps to i+1 in matrix
        # Slice matrix to get distances between bins
        d_bins = context.distance_matrix[1:, 1:]

        # Sub-select only candidates to reduce computation
        sub_d_depot = d_depot[candidate_indices]
        sub_d_bins = d_bins[np.ix_(candidate_indices, candidate_indices)]

        # Vectorized savings calculation:
        # S[i, j] = d(0, i) + d(0, j) - d(i, j)
        # We can use broadcasting for the d(0, i) + d(0, j) part
        savings = sub_d_depot[:, np.newaxis] + sub_d_depot[np.newaxis, :] - sub_d_bins

        # Ignore diagonal (i == j)
        np.fill_diagonal(savings, -1.0)

        # A bin i is selected if there exists any j such that savings[i, j] > 0
        has_saving = np.any(savings > 0, axis=1)

        mandatory_indices = candidate_indices[has_saving]

        return sorted((mandatory_indices + 1).tolist()), SearchContext.initialize(
            selection_metrics={"strategy": "SavingsSelection"}
        )
