"""
Spatial Synergy Packing Algorithm Module.

This strategy leverages spatial graph properties to amortize routing costs.
It identifies "critical" bins that must be collected (e.g., > 90% full) and
extends the obligatory collection set to include nearby bins that are
moderately full (e.g., > 60% full). By visiting these moderately full bins
opportunistically, the vehicle avoids returning to the same neighborhood
the very next day.

Attributes:
    SpatialSynergySelection(IPackingAlgorithm): Spatial-density based collection strategy.

Example:
    >>> from logic.src.policies.helpers.mandatory.selection_spatial_synergy import SpatialSynergySelection
    >>> strategy = SpatialSynergySelection()
    >>> bins = strategy.pack(context)
"""

from typing import List, Tuple

import numpy as np

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context import PackingContext, SearchContext
from logic.src.interfaces.packing import IPackingAlgorithm
from logic.src.policies.packing.base import PackingRegistry


@GlobalRegistry.register(
    PolicyTag.PACKING,
    PolicyTag.HEURISTIC,
)
@PackingRegistry.register("spatial_synergy")
class SpatialSynergySelection(IPackingAlgorithm):
    """Spatial-density based collection strategy.

    Attributes:
        None
    """

    def pack(self, context: PackingContext) -> Tuple[List[int], SearchContext]:
        """Select critical bins and their valid spatial synergies.

        Args:
            context: The selection context providing current_fill and distance_matrix.

        Returns:
            Tuple[List[int], SearchContext]: Selected bin IDs (1-based) and search context.

        Raises:
            ValueError: If ``distance_matrix`` is missing.
        """
        if context.distance_matrix is None:
            raise ValueError("SpatialSynergySelection requires a distance_matrix in the context.")

        # Extract parameters dynamically from context with robust defaults
        critical_threshold = getattr(context, "critical_threshold", 0.90)
        synergy_threshold = getattr(context, "synergy_threshold", 0.60)
        radius = getattr(context, "radius", 10.0)

        # Calculate fill ratio array
        fill_ratios = context.current_fill / context.max_fill

        # 1. Identify critically full bins (0-based indices)
        critical_indices = np.nonzero(fill_ratios > critical_threshold)[0]
        if len(critical_indices) == 0:
            return [], SearchContext.initialize(selection_metrics={"strategy": "SpatialSynergySelection"})

        # 2. Identify moderately full bins that are eligible for synergy
        moderate_indices = np.nonzero(fill_ratios > synergy_threshold)[0]
        mandatory_set = set(critical_indices.tolist())

        # 3. For each critical bin, find moderately full neighbors
        # Note: distance_matrix includes the depot at index 0.
        # Bin i corresponds to index i+1 in the distance_matrix.
        for crit_idx in critical_indices:
            dist_to_crit = context.distance_matrix[crit_idx + 1, 1:]

            # Find all bins within the specified spatial radius
            neighbors = np.nonzero(dist_to_crit <= radius)[0]

            # Intersect spatial neighbors with the moderately full condition
            synergy_bins = np.intersect1d(neighbors, moderate_indices)
            mandatory_set.update(synergy_bins.tolist())

        # Convert back to 1-based IDs for the routing engine
        return [i + 1 for i in mandatory_set], SearchContext.initialize(
            selection_metrics={"strategy": "SpatialSynergySelection"}
        )
