"""
Global wrapper for Stringing operators as a general repair heuristic.

Provides `stringing_insertion` and `stringing_profit_insertion` wrappers
that mirror `greedy_insertion`, capable of iterating over unassigned nodes
(with `expand_pool` support) and inserting them using GENIUS Stringing moves.

Supports both:
1. Deterministic p-neighborhood search (strict adherence to Gendreau et al. 1992)
2. Randomized sampling mode (for faster approximate search)

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.generalized_insertion_and_deletion.stringing_wrapper import stringing_insertion
    >>> routes = stringing_insertion(routes, removed_nodes, 1, dist_matrix, wastes, capacity)

Note:
    Search and move application routines are located in :mod:`.stringing_search`.
"""

import random
from typing import Dict, List, Optional

import numpy as np

from .stringing_search import (
    _apply_stringing_op as _apply_stringing_op,
)
from .stringing_search import (
    _evaluate_routes as _evaluate_routes,
)
from .stringing_search import (
    _try_string_insertion as _try_string_insertion,
)
from .stringing_search import (
    _try_string_insertion_deterministic as _try_string_insertion_deterministic,
)

__all__ = [
    "stringing_insertion",
    "stringing_profit_insertion",
    "stringing_insertion_wrapper",
    "_apply_stringing_op",
    "_evaluate_routes",
    "_try_string_insertion",
    "_try_string_insertion_deterministic",
]


def stringing_insertion_wrapper(  # noqa: C901
    routes: List[List[int]],
    removed_nodes: List[int],
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float = 0.0,
    C: float = 1.0,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
    profit_mode: bool = False,
    expand_pool: bool = False,
    use_alns_fallback: bool = False,
    random_us_sampling: bool = True,
    neighborhood_size: int = 5,
) -> List[List[int]]:
    """
    Core stringing insertion logic with conditional search mode.

    Args:
        routes: Current solution.
        removed_nodes: Nodes to insert.
        string_type: Stringing operator type (1-4).
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue per unit.
        C: Cost per unit distance.
        mandatory_nodes: Mandatory nodes.
        rng: Random generator.
        profit_mode: Use profit logic.
        expand_pool: Whether to expand insertion pool to all unassigned nodes.
        use_alns_fallback: Use greedy fallback if stringing fails.
        random_us_sampling: If True, use random sampling; if False, use deterministic p-neighborhoods.
        neighborhood_size: Size of p-neighborhood (for deterministic mode).

    Returns:
        List[List[int]]: Repaired solution.
    """
    if rng is None:
        rng = random.Random()

    from logic.src.policies.helpers.operators.recreate_repair.greedy import (
        greedy_insertion,
        greedy_profit_insertion,
    )

    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()

    # Establish pool of unassigned candidates
    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    # Iterate until pool is resolved
    while unassigned:
        best_routes = None
        best_value = -float("inf")
        best_node = -1

        # O(N) Pre-calculate Route Capacities
        route_loads = [sum(wastes.get(n, 0) for n in r) for r in routes]

        # Try inserting every available unassigned node into every route
        for node in unassigned:
            is_mandatory = node in mandatory_nodes_set
            node_waste = wastes.get(node, 0)

            # Evaluate opening a brand new route `[0, node, 0]`
            if profit_mode:
                cost_new = dist_matrix[0, node] + dist_matrix[node, 0]
                val_new = (node_waste * R) - (cost_new * C)
                effective_val_new = val_new + (1e9 if is_mandatory else 0)

                # Reject negative pure-profit insertions unless mandatory
                if (is_mandatory or val_new >= -1e-4) and effective_val_new > best_value:
                    best_value = effective_val_new
                    test_routes_new = [list(r) for r in routes]
                    test_routes_new.append([node])
                    best_routes = test_routes_new
                    best_node = node
            else:
                test_routes_new = [list(r) for r in routes]
                test_routes_new.append([node])
                val_new = _evaluate_routes(test_routes_new, dist_matrix)
                if val_new > best_value:
                    best_value = val_new
                    best_routes = test_routes_new
                    best_node = node

            for r_idx in range(len(routes)):
                # Early Pruning: Capacity Constraint Check BEFORE topological iterations
                current_load = route_loads[r_idx]
                if current_load + node_waste > capacity:
                    continue

                # Choose search strategy based on random_us_sampling flag
                if random_us_sampling:
                    result = _try_string_insertion(
                        routes,
                        node,
                        r_idx,
                        string_type,
                        dist_matrix,
                        wastes,
                        capacity,
                        R,
                        C,
                        rng,
                        current_load,
                        profit_mode,
                    )
                else:
                    result = _try_string_insertion_deterministic(
                        routes,
                        node,
                        r_idx,
                        string_type,
                        dist_matrix,
                        wastes,
                        capacity,
                        R,
                        C,
                        current_load,
                        neighborhood_size,
                        profit_mode,
                    )

                if result:
                    test_routes, val = result

                    if profit_mode:
                        # Bias score slightly for mandatory nodes in VRPP mode
                        effective_val = val + (1e9 if is_mandatory else 0)
                        # Reject negative pure-profit insertions unless mandatory
                        if not is_mandatory and val < -1e-4:
                            continue

                        if effective_val > best_value:
                            best_value = effective_val
                            best_routes = test_routes
                            best_node = node
                    else:
                        if val > best_value:
                            best_value = val
                            best_routes = test_routes
                            best_node = node

        # Apply best found insertion
        if best_node != -1 and best_routes is not None:
            routes = best_routes
            unassigned.remove(best_node)
        else:
            if not use_alns_fallback:
                # If unable to string properly, and fallback is disabled, abandon unassigned nodes
                break

            # Fallback for remaining nodes that couldn't be strung (too small routes, complex constraints)
            if profit_mode:
                routes = greedy_profit_insertion(
                    routes,
                    unassigned,
                    dist_matrix,
                    wastes,
                    capacity,
                    R,
                    C,
                    mandatory_nodes,
                    expand_pool=False,  # Already isolated to unassigned pool
                )
            else:
                routes = greedy_insertion(
                    routes,
                    unassigned,
                    dist_matrix,
                    wastes,
                    capacity,
                    mandatory_nodes=mandatory_nodes,
                    expand_pool=False,
                )
            break  # Greedy fallback completely drains the unassigned array

    return routes


# ==== Public Wrapper API ====


def stringing_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
    expand_pool: bool = False,
    use_alns_fallback: bool = False,
    random_us_sampling: bool = True,
    neighborhood_size: int = 5,
) -> List[List[int]]:
    """
    Standard CVRP Stringing Repair.

    Args:
        routes: Current routes.
        removed_nodes: Nodes to insert.
        string_type: Stringing type.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        mandatory_nodes: Mandatory nodes.
        rng: Random generator.
        expand_pool: Expand pool flag.
        use_alns_fallback: Greedy fallback flag.
        random_us_sampling: Sampling mode flag.
        neighborhood_size: Neighborhood size.

    Returns:
        List[List[int]]: Repaired routes.
    """
    return stringing_insertion_wrapper(
        routes,
        removed_nodes,
        string_type,
        dist_matrix,
        wastes,
        capacity,
        mandatory_nodes=mandatory_nodes,
        rng=rng,
        profit_mode=False,
        expand_pool=expand_pool,
        use_alns_fallback=use_alns_fallback,
        random_us_sampling=random_us_sampling,
        neighborhood_size=neighborhood_size,
    )


def stringing_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
    expand_pool: bool = False,
    use_alns_fallback: bool = False,
    random_us_sampling: bool = True,
    neighborhood_size: int = 5,
) -> List[List[int]]:
    """
    VRPP Cost-Aware Stringing Repair.

    Args:
        routes: Current routes.
        removed_nodes: Nodes to insert.
        string_type: Stringing type.
        dist_matrix: Distance matrix.
        wastes: Node wastes.
        capacity: Vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        mandatory_nodes: Mandatory nodes.
        rng: Random generator.
        expand_pool: Expand pool flag.
        use_alns_fallback: Greedy fallback flag.
        random_us_sampling: Sampling mode flag.
        neighborhood_size: Neighborhood size.

    Returns:
        List[List[int]]: Repaired routes.
    """
    # Feedback implementation: Keep Variants I, II, and III purely structural
    profit_mode = string_type not in (1, 2, 3)
    return stringing_insertion_wrapper(
        routes,
        removed_nodes,
        string_type,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        mandatory_nodes=mandatory_nodes,
        rng=rng,
        profit_mode=profit_mode,
        expand_pool=expand_pool,
        use_alns_fallback=use_alns_fallback,
        random_us_sampling=random_us_sampling,
        neighborhood_size=neighborhood_size,
    )
