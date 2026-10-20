"""
LKH-2 (Lin-Kernighan-Helsgaun 2009) Route Improver.

Attributes:
    LinKernighanHelsgaunTwoRouteImprover: Main class for LKH-2 based improvement.

Example:
    >>> from logic.src.policies.route_improvement.lkh2 import LinKernighanHelsgaunTwoRouteImprover
    >>> improver = LinKernighanHelsgaunTwoRouteImprover(config=cfg)
    >>> refined_tour, metrics = improver.process(tour, max_iterations=200)
"""

from typing import Any, List, Tuple

import numpy as np

from logic.src.enums import PolicyTag
from logic.src.interfaces import IRouteImprovement
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.policies.helpers.operators.search_heuristics.lin_kernighan_helsgaun_two import (
    solve_lkh2,
)
from logic.src.utils.registry import ALL

from .base import RouteImproverRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@ALL.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.HEURISTIC,
)
@RouteImproverRegistry.register("lkh2")
class LinKernighanHelsgaunTwoRouteImprover(IRouteImprovement):
    """LKH-2 route improver.

    Handles VRPP subset routes by extracting a dense sub-matrix containing only
    the visited nodes, running LKH-2 on this sub-problem, and mapping the results
    back to the original node IDs.

    Attributes:
        config (Dict[str, Any]): Internal configuration state.

    Example:
        >>> improver = LinKernighanHelsgaunTwoRouteImprover(config=cfg)
        >>> refined_tour, metrics = improver.process(tour, max_iterations=200)
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """
        Apply LKH-2 refinement to a tour using sub-matrix extraction.

        For VRPP instances where only a subset of nodes are visited, this method
        creates a dense sub-problem by extracting only the relevant rows and
        columns from the full distance matrix.  This prevents index out-of-bounds
        errors and eliminates performance bottlenecks from hallucinated nodes
        during tour merging.

        Args:
            tour:      The initial tour to refine (list of node IDs from the full
                       problem).
            kwargs:    Context containing:
                       - distance_matrix: Distance matrix.
                       - max_iterations: Maximum iterations.
                       - max_k: Maximum k for LKH.
                       - population_size: Population size.
                       - n_candidates: Number of candidates.
                       - seed: Random seed.

        Returns:
            Tuple[List[int], ImprovementMetrics]: The refined tour with original
            node IDs and a metrics dict.
        """
        _algo = "LinKernighanHelsgaunTwoRouteImprover"

        # 1. Early exits & edge cases
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None:
            return tour, {"algorithm": _algo}

        dm = to_numpy(distance_matrix)

        if not tour:
            return tour, {"algorithm": _algo}

        # 2. Split tour into per-trip sub-problems
        routes = split_tour(tour)
        if not routes:
            return tour, {"algorithm": _algo}

        # 3. Solver parameters
        max_iterations = kwargs.get("max_iterations", self.config.get("max_iterations", 200))
        max_k = kwargs.get("max_k", self.config.get("max_k", 5))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        refined_routes = []
        for trip in routes:
            if len(trip) < 2:
                refined_routes.append(trip)
                continue

            # 4. Node mapping: always include depot (0)
            unique_nodes = [0] + sorted(set(trip))
            node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}

            # 5. Extract dense sub-matrix
            sub_matrix = dm[np.ix_(unique_nodes, unique_nodes)]

            # 6. Translate trip to dense indices
            sub_tour_indices = [node_to_idx[0]] + [node_to_idx[node] for node in trip] + [node_to_idx[0]]

            try:
                optimized_indices, _, _ = solve_lkh2(
                    sub_matrix,
                    initial_tour=sub_tour_indices,
                    max_iterations=max_iterations,
                    max_k=max_k,
                    seed=seed,
                )
                # Map back and strip depots (assemble_tour re-adds them)
                refined_trip = [unique_nodes[idx] for idx in optimized_indices if unique_nodes[idx] != 0]
                refined_routes.append(refined_trip)
            except Exception:
                # Keep the original trip if LKH-2 fails for any reason
                refined_routes.append(trip)

        return assemble_tour(refined_routes), {"algorithm": _algo}
