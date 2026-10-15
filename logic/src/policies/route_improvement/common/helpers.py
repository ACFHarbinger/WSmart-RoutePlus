"""
Shared helpers for route improvers.

Kept deliberately small and stateless so route improvers can copy import
this module without coupling themselves to each other.

Attributes:
    None

Example:
    >>> from logic.src.policies.route_improvement.common.helpers import split_tour
    >>> routes = split_tour([0, 1, 2, 0, 3, 4, 0])
    >>> print(routes)
    [[1, 2], [3, 4]]
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


def resolve_mandatory_nodes(
    kwargs: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[List[int]]:
    """Resolve the list of mandatory-node IDs from the route improver context.

    Checks keys in priority order:
        1. kwargs["mandatory_nodes"] — explicit caller override
        2. kwargs["mandatory"]         — packing action output
        3. config["mandatory_nodes"] — config file default

    Args:
        kwargs: Keyword arguments passed to the process method.
        config: Internal configuration dictionary of the improver.

    Returns:
        Optional[List[int]]: List of mandatory node IDs, or None if not found.
    """
    if "mandatory_nodes" in kwargs and kwargs["mandatory_nodes"] is not None:
        return list(kwargs["mandatory_nodes"])
    if "mandatory" in kwargs and kwargs["mandatory"] is not None:
        return list(kwargs["mandatory"])
    if "mandatory_nodes" in config and config["mandatory_nodes"] is not None:
        return list(config["mandatory_nodes"])
    return None


def upgrade_repair_op_to_profit(repair_op: str, revenue_kg: float, cost_per_km: float) -> str:
    """Return the profit-aware variant of a repair operator.

    Args:
        repair_op: Name of the repair operator.
        revenue_kg: Revenue per kg of waste.
        cost_per_km: Cost per km of distance.

    Returns:
        str: The updated operator name with "_profit" suffix if applicable.
    """
    if (revenue_kg > 0 or cost_per_km > 0) and "profit" not in repair_op:
        return repair_op + "_profit"
    return repair_op


def to_numpy(distance_matrix: Any) -> np.ndarray:
    """Safely convert distance matrix to numpy array, handling Torch tensors.

    Args:
        distance_matrix: Input distance matrix (array, tensor, or list).

    Returns:
        np.ndarray: The distance matrix as a numpy array.
    """
    if isinstance(distance_matrix, np.ndarray):
        return distance_matrix
    if hasattr(distance_matrix, "cpu"):
        return distance_matrix.cpu().numpy()
    return np.array(distance_matrix)


def split_tour(tour: Sequence[int]) -> List[List[int]]:
    """Split a multi-trip tour separated by depot 0s into per-trip lists.

    Args:
        tour: A sequence of node IDs representing a tour.

    Returns:
        List[List[int]]: A list of routes, where each route is a list of node IDs.
    """
    routes: List[List[int]] = []
    current: List[int] = []
    for node in tour:
        if node == 0:
            if current:
                routes.append(current)
                current = []
        else:
            current.append(node)
    if current:
        routes.append(current)
    return routes


def assemble_tour(routes: Sequence[Sequence[int]]) -> List[int]:
    """Reassemble per-trip routes into a depot-separated tour.

    Args:
        routes: Sequence of routes to assemble.

    Returns:
        List[int]: A single tour starting and ending with the depot (0).
    """
    result: List[int] = [0]
    for r in routes:
        if r:
            result.extend(r)
            result.append(0)
    return result


def route_distance(route: Sequence[int], dm: np.ndarray) -> float:
    """Length of a single route, including depot-out and depot-return legs.

    Args:
        route: Sequence of node IDs in the route.
        dm: Distance matrix.

    Returns:
        float: The total distance of the route.
    """
    if not route:
        return 0.0
    total = float(dm[0, route[0]]) + float(dm[route[-1], 0])
    for i in range(len(route) - 1):
        total += float(dm[route[i], route[i + 1]])
    return total


def tour_distance(routes: Iterable[Sequence[int]], dm: np.ndarray) -> float:
    """Total distance over all routes.

    Args:
        routes: Collection of routes.
        dm: Distance matrix.

    Returns:
        float: Total distance of all routes combined.
    """
    return sum(route_distance(r, dm) for r in routes)


def route_load(route: Sequence[int], wastes: Dict[int, float]) -> float:
    """Sum of waste masses for a single route.

    Args:
        route: Sequence of node IDs in the route.
        wastes: Dictionary mapping node IDs to their waste mass.

    Returns:
        float: Total load of the route.
    """
    return sum(float(wastes.get(n, 0.0)) for n in route)
