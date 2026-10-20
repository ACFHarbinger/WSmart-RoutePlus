"""
Global wrapper for Unstringing operators as a general destroy/removal heuristic.

Provides `unstringing_removal` and `unstringing_profit_removal` wrappers
that remove nodes by employing GENIUS Unstringing moves to gracefully extract
nodes from existing routes while rewiring them to minimize cost bumps.

Supports both:
1. Deterministic p-neighborhood search (strict adherence to Gendreau et al. 1992)
2. Randomized sampling mode (for faster approximate search)

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.unstringing_wrapper import unstringing_removal
    >>> routes, removed = unstringing_removal(routes, 1, 1, dist_matrix)

Note:
    Search and move application routines are located in :mod:`.unstringing_search`.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .unstringing_search import (
    _apply_unstring_op as _apply_unstring_op,
)
from .unstringing_search import (
    _evaluate_routes as _evaluate_routes,
)
from .unstringing_search import (
    _try_unstring_removal as _try_unstring_removal,
)
from .unstringing_search import (
    _try_unstring_removal_deterministic as _try_unstring_removal_deterministic,
)

__all__ = [
    "unstringing_removal",
    "unstringing_profit_removal",
    "unstringing_removal_wrapper",
    "_apply_unstring_op",
    "_evaluate_routes",
    "_try_unstring_removal",
    "_try_unstring_removal_deterministic",
]


def unstringing_removal_wrapper(  # noqa: C901
    routes: List[List[int]],
    n_remove: int,
    unstring_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 0.0,
    C: float = 1.0,
    rng: Optional[random.Random] = None,
    profit_mode: bool = False,
    target_node: Optional[int] = None,
    use_alns_fallback: bool = False,
    random_us_sampling: bool = True,
    neighborhood_size: int = 5,
) -> Tuple[List[List[int]], List[int]]:
    """
    Core unstringing removal logic with conditional search mode.

    Args:
        routes: Current solution.
        n_remove: Number of nodes to remove.
        unstring_type: Unstringing operator type (1-4).
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        R: Revenue per unit.
        C: Cost per unit distance.
        rng: Random generator.
        profit_mode: Use profit logic.
        target_node: If provided, only try removing this specific node.
        use_alns_fallback: Use worst removal if unstringing fails.
        random_us_sampling: If True, use random sampling; if False, use deterministic p-neighborhoods.
        neighborhood_size: Size of p-neighborhood (for deterministic mode).

    Returns:
        Tuple[List[List[int]], List[int]]: (repaired_solution, removed_nodes).
    """
    if rng is None:
        rng = random.Random()

    from logic.src.policies.helpers.operators.destroy_ruin.worst import (
        worst_profit_removal,
        worst_removal,
    )

    removed_nodes = []
    for _ in range(n_remove):
        best_routes = None
        best_value = -float("inf")
        best_node = -1

        # Consider all currently assigned nodes
        for r_idx, route in enumerate(routes):
            if len(route) < 3:
                continue
            for n_idx, node in enumerate(route):
                if target_node is not None and node != target_node:
                    continue

                # Choose search strategy based on random_us_sampling flag
                if random_us_sampling:
                    result = _try_unstring_removal(
                        routes, r_idx, n_idx, unstring_type, dist_matrix, wastes, R, C, rng, profit_mode
                    )
                else:
                    result = _try_unstring_removal_deterministic(
                        routes, r_idx, n_idx, unstring_type, dist_matrix, wastes, R, C, neighborhood_size, profit_mode
                    )

                if result:
                    test_routes, val = result
                    if val > best_value:
                        best_value = val
                        best_routes = test_routes
                        best_node = node

        if best_node != -1 and best_routes is not None:
            routes = best_routes
            removed_nodes.append(best_node)
            routes = [r for r in routes if r]
            if target_node is not None and best_node == target_node:
                break
        else:
            if not use_alns_fallback:
                # Gracefully ignore if GENI fails without ALNS fallback
                break

            # Fallback for remaining nodes that couldn't be unstrung
            nodes_to_remove = min(1, sum(len(r) for r in routes))
            if nodes_to_remove == 0:
                break

            if profit_mode:
                routes, rem = worst_profit_removal(
                    routes,
                    nodes_to_remove,
                    dist_matrix,
                    wastes,
                    R,
                    C,
                )
            else:
                routes, rem = worst_removal(
                    routes,
                    nodes_to_remove,
                    dist_matrix,
                )
            removed_nodes.extend(rem)
            if target_node is not None and target_node in rem:
                break

    return routes, removed_nodes


# ==== Public Wrapper API ====


def unstringing_removal(
    routes: List[List[int]],
    n_remove: int,
    unstring_type: int,
    dist_matrix: np.ndarray,
    rng: Optional[random.Random] = None,
    target_node: Optional[int] = None,
    use_alns_fallback: bool = False,
    random_us_sampling: bool = True,
    neighborhood_size: int = 5,
) -> Tuple[List[List[int]], List[int]]:
    """
    Standard CVRP Unstringing Removal.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        unstring_type: Unstringing type.
        dist_matrix: Distance matrix.
        rng: Random generator.
        target_node: Target node.
        use_alns_fallback: Fallback flag.
        random_us_sampling: Sampling mode flag.
        neighborhood_size: Neighborhood size.

    Returns:
        Tuple[List[List[int]], List[int]]: (new_routes, removed_nodes).
    """
    return unstringing_removal_wrapper(
        routes=routes,
        n_remove=n_remove,
        unstring_type=unstring_type,
        dist_matrix=dist_matrix,
        wastes={},
        rng=rng,
        profit_mode=False,
        target_node=target_node,
        use_alns_fallback=use_alns_fallback,
        random_us_sampling=random_us_sampling,
        neighborhood_size=neighborhood_size,
    )


def unstringing_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    unstring_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[random.Random] = None,
    target_node: Optional[int] = None,
    use_alns_fallback: bool = False,
    random_us_sampling: bool = True,
    neighborhood_size: int = 5,
) -> Tuple[List[List[int]], List[int]]:
    """
    VRPP Cost-Aware Unstringing Removal.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        unstring_type: Unstringing type.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        R: Revenue multiplier.
        C: Cost multiplier.
        rng: Random generator.
        target_node: Target node.
        use_alns_fallback: Fallback flag.
        random_us_sampling: Sampling mode flag.
        neighborhood_size: Neighborhood size.

    Returns:
        Tuple[List[List[int]], List[int]]: (new_routes, removed_nodes).
    """
    # Feedback implementation: Keep Variants I, II, and III purely structural
    profit_mode = unstring_type not in (1, 2, 3)
    return unstringing_removal_wrapper(
        routes=routes,
        n_remove=n_remove,
        unstring_type=unstring_type,
        dist_matrix=dist_matrix,
        wastes=wastes,
        R=R,
        C=C,
        rng=rng,
        profit_mode=profit_mode,
        target_node=target_node,
        use_alns_fallback=use_alns_fallback,
        random_us_sampling=random_us_sampling,
        neighborhood_size=neighborhood_size,
    )
