"""
Composite destroy-and-repair low-level heuristics (LLHs) for trajectory-based
metaheuristics.

Each LLH pairs one removal operator from ``destroy_ruin`` with one insertion
operator from ``recreate_repair`` and reinserts the removed nodes. These
composites are shared by the ILS and VNS solvers (and usable by any other
descent-style metaheuristic) so the combination logic lives in one place.

Attributes:
    llh_random_greedy: Random removal + greedy insertion.
    llh_worst_regret_2: Worst removal + regret-2 insertion.
    llh_cluster_greedy: Cluster removal + greedy insertion.
    llh_worst_greedy: Worst removal + greedy insertion.
    llh_random_regret_2: Random removal + regret-2 insertion.
    routes_total_distance: Total depot-returned distance of a route set.
    routes_net_profit: Net profit (revenue - travel cost) of a route set.
    build_greedy_initial_routes: Greedy initial solution construction.

Example:
    >>> from random import Random
    >>> routes = llh_random_greedy(routes, 2, dist_matrix, wastes, capacity, R, C,
    ...                            mandatory_nodes, expand_pool=True,
    ...                            profit_aware=False, rng=Random(0))
"""

from random import Random
from typing import Dict, List, Optional

import numpy as np

from logic.src.policies.helpers.operators.destroy_ruin import (
    cluster_removal,
    random_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_insertion,
    greedy_profit_insertion,
    regret_2_insertion,
    regret_2_profit_insertion,
)
from logic.src.policies.helpers.operators.solution_initialization import build_greedy_routes


def _insert_greedy(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    expand_pool: bool,
    profit_aware: bool,
) -> List[List[int]]:
    """Reinsert nodes with greedy insertion, optionally profit-aware."""
    if profit_aware:
        return greedy_profit_insertion(
            routes, removed_nodes, dist_matrix, wastes, capacity, R, C, mandatory_nodes, expand_pool
        )
    return greedy_insertion(
        routes, removed_nodes, dist_matrix, wastes, capacity, mandatory_nodes=mandatory_nodes, expand_pool=expand_pool
    )


def _insert_regret_2(
    routes: List[List[int]],
    removed_nodes: List[int],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    expand_pool: bool,
    profit_aware: bool,
) -> List[List[int]]:
    """Reinsert nodes with regret-2 insertion, optionally profit-aware."""
    if profit_aware:
        return regret_2_profit_insertion(
            routes, removed_nodes, dist_matrix, wastes, capacity, R, C, mandatory_nodes, expand_pool
        )
    return regret_2_insertion(
        routes, removed_nodes, dist_matrix, wastes, capacity, mandatory_nodes=mandatory_nodes, expand_pool=expand_pool
    )


def _remove_worst(
    routes: List[List[int]],
    n: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    profit_aware: bool,
) -> tuple:
    """Remove nodes with worst removal, optionally profit-aware."""
    if profit_aware:
        return worst_profit_removal(routes, n, dist_matrix, wastes, R, C)
    return worst_removal(routes, n, dist_matrix)


def llh_random_greedy(
    routes: List[List[int]],
    n: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    profit_aware: bool = False,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """Random removal followed by greedy reinsertion.

    Args:
        routes: Current routes.
        n: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Mapping of node indices to waste amounts.
        capacity: Vehicle capacity.
        R: Revenue per unit collected.
        C: Cost per unit distance.
        mandatory_nodes: Nodes that must be visited.
        expand_pool: If True, consider all unvisited nodes for insertion.
        profit_aware: Use profit-aware removal/insertion variants.
        rng: Random number generator.

    Returns:
        Repaired routes.
    """
    partial, removed = random_removal(routes, n, rng)
    return _insert_greedy(
        partial, removed, dist_matrix, wastes, capacity, R, C, mandatory_nodes or [], expand_pool, profit_aware
    )


def llh_worst_regret_2(
    routes: List[List[int]],
    n: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    profit_aware: bool = False,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """Worst removal followed by regret-2 reinsertion.

    Args:
        routes: Current routes.
        n: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Mapping of node indices to waste amounts.
        capacity: Vehicle capacity.
        R: Revenue per unit collected.
        C: Cost per unit distance.
        mandatory_nodes: Nodes that must be visited.
        expand_pool: If True, consider all unvisited nodes for insertion.
        profit_aware: Use profit-aware removal/insertion variants.
        rng: Random number generator (unused to preserve the legacy entropy fallback).

    Returns:
        Repaired routes.
    """
    del rng  # Preserve the legacy unseeded worst-removal call.; kept for a uniform LLH signature
    partial, removed = _remove_worst(routes, n, dist_matrix, wastes, R, C, profit_aware)
    return _insert_regret_2(
        partial, removed, dist_matrix, wastes, capacity, R, C, mandatory_nodes or [], expand_pool, profit_aware
    )


def llh_cluster_greedy(
    routes: List[List[int]],
    n: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    profit_aware: bool = False,
    rng: Optional[Random] = None,
    nodes: Optional[List[int]] = None,
) -> List[List[int]]:
    """Cluster removal followed by greedy reinsertion.

    Args:
        routes: Current routes.
        n: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Mapping of node indices to waste amounts.
        capacity: Vehicle capacity.
        R: Revenue per unit collected.
        C: Cost per unit distance.
        mandatory_nodes: Nodes that must be visited.
        expand_pool: If True, consider all unvisited nodes for insertion.
        profit_aware: Use profit-aware removal/insertion variants.
        rng: Random number generator.
        nodes: Customer node indices (defaults to ``range(1, n_nodes + 1)``).

    Returns:
        Repaired routes.
    """
    node_pool = nodes if nodes is not None else list(range(1, len(dist_matrix)))
    partial, removed = cluster_removal(routes, n, dist_matrix, node_pool, rng)
    return _insert_greedy(
        partial, removed, dist_matrix, wastes, capacity, R, C, mandatory_nodes or [], expand_pool, profit_aware
    )


def llh_worst_greedy(
    routes: List[List[int]],
    n: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    profit_aware: bool = False,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """Worst removal followed by greedy reinsertion.

    Args:
        routes: Current routes.
        n: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Mapping of node indices to waste amounts.
        capacity: Vehicle capacity.
        R: Revenue per unit collected.
        C: Cost per unit distance.
        mandatory_nodes: Nodes that must be visited.
        expand_pool: If True, consider all unvisited nodes for insertion.
        profit_aware: Use profit-aware removal/insertion variants.
        rng: Random number generator (unused to preserve the legacy entropy fallback).

    Returns:
        Repaired routes.
    """
    del rng  # Preserve the legacy unseeded worst-removal call.; kept for a uniform LLH signature
    partial, removed = _remove_worst(routes, n, dist_matrix, wastes, R, C, profit_aware)
    return _insert_greedy(
        partial, removed, dist_matrix, wastes, capacity, R, C, mandatory_nodes or [], expand_pool, profit_aware
    )


def llh_random_regret_2(
    routes: List[List[int]],
    n: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    expand_pool: bool = True,
    profit_aware: bool = False,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """Random removal followed by regret-2 reinsertion.

    Args:
        routes: Current routes.
        n: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Mapping of node indices to waste amounts.
        capacity: Vehicle capacity.
        R: Revenue per unit collected.
        C: Cost per unit distance.
        mandatory_nodes: Nodes that must be visited.
        expand_pool: If True, consider all unvisited nodes for insertion.
        profit_aware: Use profit-aware removal/insertion variants.
        rng: Random number generator.

    Returns:
        Repaired routes.
    """
    partial, removed = random_removal(routes, n, rng)
    return _insert_regret_2(
        partial, removed, dist_matrix, wastes, capacity, R, C, mandatory_nodes or [], expand_pool, profit_aware
    )


def routes_total_distance(routes: List[List[int]], dist_matrix: np.ndarray) -> float:
    """Total depot-returned travel distance of a route set.

    Args:
        routes: Routing sequences (node indices, depot is index 0).
        dist_matrix: Distance matrix.

    Returns:
        Total distance.
    """
    total = 0.0
    for route in routes:
        if not route:
            continue
        total += dist_matrix[0][route[0]]
        for k in range(len(route) - 1):
            total += dist_matrix[route[k]][route[k + 1]]
        total += dist_matrix[route[-1]][0]
    return total


def routes_net_profit(
    routes: List[List[int]],
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
) -> float:
    """Net profit (revenue - travel cost) of a route set.

    Args:
        routes: Routing sequences (node indices, depot is index 0).
        dist_matrix: Distance matrix.
        wastes: Mapping of node indices to waste amounts.
        R: Revenue per unit collected.
        C: Cost per unit distance.

    Returns:
        Net profit.
    """
    if not routes:
        return 0.0
    rev = sum(wastes.get(n, 0.0) * R for r in routes for n in r)
    return rev - routes_total_distance(routes, dist_matrix) * C


def build_greedy_initial_routes(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[Random] = None,
) -> List[List[int]]:
    """Construct an initial solution with the greedy construction heuristic.

    Args:
        dist_matrix: Distance matrix.
        wastes: Mapping of node indices to waste amounts.
        capacity: Vehicle capacity.
        R: Revenue per unit collected.
        C: Cost per unit distance.
        mandatory_nodes: Nodes that must be visited.
        rng: Random number generator.

    Returns:
        Initial routes.
    """
    return build_greedy_routes(
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        R=R,
        C=C,
        mandatory_nodes=mandatory_nodes,
        rng=rng,
    )
