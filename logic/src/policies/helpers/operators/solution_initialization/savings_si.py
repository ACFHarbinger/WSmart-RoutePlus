"""
Clarke-Wright Savings Algorithm for VRPP initialization.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.solution_initialization.savings_si import build_savings_routes
    >>> routes = build_savings_routes(dist_matrix, wastes, capacity, R, C)

Reference:
    Clarke, G., & Wright, J. W. (1964). Scheduling of vehicles from a central
    depot to a number of delivery points. Operations Research, 12(4), 568-581.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.utils.policy.routes import (
    prune_unprofitable_routes,
)


def _compute_savings(eligible: List[int], dist_matrix: np.ndarray) -> List[Tuple[float, int, int]]:
    """
    Compute and sort all positive savings pairs.

    Args:
        eligible: List of eligible nodes.
        dist_matrix: Distance matrix.

    Returns:
        List[Tuple[float, int, int]]: Sorted list of (saving, node_i, node_j).
    """
    savings = []
    for idx, i in enumerate(eligible):
        for j in eligible[idx + 1 :]:
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            if s > 0:
                savings.append((s, i, j))
    savings.sort(reverse=True)
    return savings


def _try_merge(
    s: float,
    i: int,
    j: int,
    route_of: Dict[int, List[int]],
    route_key: Dict[int, int],
    load_of: Dict[int, float],
    capacity: float,
) -> bool:
    """
    Attempt to merge routes containing i and j.

    Args:
        s: Savings value.
        i: First node.
        j: Second node.
        route_of: Node to route mapping.
        route_key: Node to route identifier mapping.
        load_of: Route identifier to load mapping.
        capacity: Vehicle capacity.

    Returns:
        bool: True if merged, False otherwise.
    """
    ki, kj = route_key[i], route_key[j]
    if ki == kj:
        return False

    ri, rj = route_of[ki], route_of[kj]
    if load_of[ki] + load_of[kj] > capacity:
        return False

    merged = None
    if ri[-1] == i and rj[0] == j:
        merged = ri + rj
    elif ri[-1] == i and rj[-1] == j:
        merged = ri + list(reversed(rj))
    elif ri[0] == i and rj[0] == j:
        merged = list(reversed(ri)) + rj
    elif ri[0] == i and rj[-1] == j:
        merged = list(reversed(ri)) + list(reversed(rj))

    if merged is None:
        return False

    new_key = merged[0]
    new_load = load_of[ki] + load_of[kj]
    for node in merged:
        route_of[node] = merged
        route_key[node] = new_key
        load_of[node] = new_load
    return True


def build_savings_routes(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
) -> List[List[int]]:
    """
    Build routes using the Clarke-Wright Savings Algorithm.

    Args:
        dist_matrix: (N+1)x(N+1) distance matrix.
        wastes: {node_id → fill level}.
        capacity: Vehicle capacity Q.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        mandatory_nodes: Nodes that must be visited.

    Returns:
        List[List[int]]: List of routes (depot excluded).
    """
    mandatory_set = set(mandatory_nodes or [])
    nodes = list(range(1, len(dist_matrix)))
    eligible = [
        i for i in nodes if i in mandatory_set or wastes.get(i, 0.0) * R >= (dist_matrix[0, i] + dist_matrix[i, 0]) * C
    ]

    if not eligible:
        return []

    savings = _compute_savings(eligible, dist_matrix)
    route_of = {i: [i] for i in eligible}
    load_of = {i: wastes.get(i, 0.0) for i in eligible}
    route_key = {i: i for i in eligible}

    for s, i, j in savings:
        _try_merge(s, i, j, route_of, route_key, load_of, capacity)

    seen_keys = set()
    routes = []
    for i in eligible:
        k = route_key[i]
        if k not in seen_keys:
            seen_keys.add(k)
            routes.append(route_of[k])

    return prune_unprofitable_routes(routes, dist_matrix, wastes, R, C, mandatory_set)
