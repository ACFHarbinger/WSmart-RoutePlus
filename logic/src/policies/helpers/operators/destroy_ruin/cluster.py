"""
Profit-Similarity and Cluster Removal Operator Module.

This module implements removal heuristics based on spatial and attribute similarity.
Includes standard cluster removal (spatial) and profit-similarity removal (VRPP).

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.destroy.cluster import cluster_removal
    >>> routes, removed = cluster_removal(routes, n_remove=5, dist_matrix=d, nodes=all_nodes)
    >>> from logic.src.policies.helpers.operators.destroy.cluster import cluster_profit_removal
    >>> routes, removed = cluster_profit_removal(routes, n_remove=5, dist_matrix=d,
    ...                                              wastes=w, R=1.0, C=1.0)
"""

from random import Random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def _kruskal_two_clusters(route: List[int], dist_matrix: np.ndarray, d_max: float) -> Tuple[List[int], List[int]]:
    """Split a route into two clusters via Kruskal's MST (Pisinger & Ropke 2007 §5.1.4).

    Builds the MST by adding edges in ascending r_ij = d(i,j)/d_max order and
    terminates after adding n-2 edges, leaving exactly two connected components.

    Args:
        route: Node IDs to partition.
        dist_matrix: Distance matrix.
        d_max: Global maximum distance for normalization.

    Returns:
        Two lists of node IDs representing the two clusters.
    """
    n = len(route)
    if n <= 1:
        return route[:], []
    if n == 2:
        return [route[0]], [route[1]]

    # All pairwise edges sorted by normalized relatedness r_ij = d(i,j) / d_max
    edges: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            u, v = route[i], route[j]
            r = float(dist_matrix[u, v]) / d_max if d_max > 0.0 else 0.0
            edges.append((r, i, j))
    edges.sort()

    # Union-Find with path halving and union by rank
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        """
        Finds the root of the element x with path halving.

        Args:
            x: Element to find.

        Returns:
            The root of element x.
        """
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        """
        Unions the sets containing x and y by rank.

        Args:
            x: First element.
            y: Second element.
        """
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # Add exactly n-2 edges → two components remain
    edges_added = 0
    for _, i, j in edges:
        if find(i) != find(j):
            union(i, j)
            edges_added += 1
            if edges_added == n - 2:
                break

    root_a = find(0)
    comp_a = [route[i] for i in range(n) if find(i) == root_a]
    comp_b = [route[i] for i in range(n) if find(i) != root_a]
    return comp_a, comp_b


def _find_cross_route_neighbor(
    pivot: int,
    pivot_route_idx: int,
    routes: List[List[int]],
    removed_set: Set[int],
    dist_matrix: np.ndarray,
) -> Optional[Tuple[int, int]]:
    """Return (best_node, route_idx) of the closest unremoved node from a different route.

    Args:
        pivot: The pivot node ID used as reference for distance comparisons.
        pivot_route_idx: Index of the route containing the pivot node.
        routes: Current solution (list of routes).
        removed_set: Set of already-removed node IDs to skip.
        dist_matrix: Distance matrix including depot at index 0.

    Returns:
        A tuple of (best_node, route_idx), or None if no eligible node exists.
    """
    best_node: Optional[int] = None
    best_dist: float = float("inf")
    best_route_idx: int = -1
    for r_idx, route in enumerate(routes):
        if r_idx == pivot_route_idx:
            continue
        for node in route:
            if node in removed_set:
                continue
            d = float(dist_matrix[pivot, node])
            if d < best_dist:
                best_dist = d
                best_node = node
                best_route_idx = r_idx
    if best_node is None:
        return None
    return best_node, best_route_idx


def cluster_removal(  # noqa: C901
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    nodes: List[int],
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove clusters of nodes using Kruskal's MST partitioning.

    **Paper Reference**: Pisinger & Ropke (2007), §5.1.4 — *Cluster Removal*.

    Implements the algorithm as described in the paper:

    1. Pick a random route and partition it into two clusters by running Kruskal's
       minimum spanning tree algorithm on the route's relatedness graph
       (edge weight ``r_ij = d(i,j) / d_max``), terminating when two connected
       components remain (i.e., after adding ``|route| - 2`` edges).
    2. Randomly select one of the two clusters and remove its nodes.
    3. If fewer than ``q`` nodes have been removed, pick a random already-removed
       node as pivot, find the most-related unremoved node from a **different**
       route (minimum ``d(pivot, node)``), and partition that route into two
       clusters using the same Kruskal procedure.  Add one of the two clusters.
    4. Repeat step 3 until ``q`` nodes have been removed.

    Args:
        routes: Current routes.
        n_remove: Target number of nodes to remove.
        dist_matrix: Distance/cost matrix with depot at index 0.
        nodes: All node IDs (unused; retained for API compatibility).
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if not any(routes) or n_remove <= 0:
        return routes, []

    if rng is None:
        rng = Random()

    d_max = float(dist_matrix.max()) if dist_matrix.size > 0 else 1.0
    if d_max == 0.0:
        d_max = 1.0

    # Original route-membership map (stable; never updated during the loop)
    node_to_route_idx: Dict[int, int] = {node: r_idx for r_idx, route in enumerate(routes) for node in route}

    removed_set: Set[int] = set()

    non_empty = [r for r in routes if r]
    if not non_empty:
        return routes, []

    # --- Phase 1: partition one randomly chosen route into two clusters ---
    start_route = rng.choice(non_empty)
    comp_a, comp_b = _kruskal_two_clusters(start_route, dist_matrix, d_max)
    chosen: List[int] = rng.choice([comp_a, comp_b]) if (comp_a and comp_b) else (comp_a or comp_b)
    removed_set.update(chosen)

    # --- Phase 2: extend across routes until n_remove nodes collected ---
    while len(removed_set) < n_remove:
        pivot: int = rng.choice(list(removed_set))
        pivot_route_idx: int = node_to_route_idx.get(pivot, -1)

        result = _find_cross_route_neighbor(pivot, pivot_route_idx, routes, removed_set, dist_matrix)
        if result is None:
            break  # No unremoved nodes remain in any other route
        best_node, target_route_idx = result

        # Partition the available (non-removed) nodes of that route
        available: List[int] = [n for n in routes[target_route_idx] if n not in removed_set]
        if not available:
            break

        comp_a, comp_b = _kruskal_two_clusters(available, dist_matrix, d_max)
        chosen = rng.choice([comp_a, comp_b]) if (comp_a and comp_b) else (comp_a or comp_b)
        removed_set.update(chosen)

    # --- Apply removals ---
    final_removed: List[int] = []
    modified_routes: List[List[int]] = []
    for r in routes:
        new_route: List[int] = []
        for node in r:
            if node in removed_set:
                final_removed.append(node)
            else:
                new_route.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed


def cluster_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """
    Remove a group of nodes based on similarity in their profit contributions.

    Unlike standard Cluster Removal which uses spatial proximity (Pisinger & Ropke 2007),
    this is a novel VRPP adaptation that destroys routes by targeting nodes with
    similar marginal profit levels, effectively isolating regions of equivalent
    unprofitability for the search to re-evaluate.

    Args:
        routes: Current routes.
        n_remove: Number of nodes to remove.
        dist_matrix: Distance matrix.
        wastes: Waste/profit values for each node.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        rng: Random number generator.

    Returns:
        Tuple[List[List[int]], List[int]]: Partial routes and list of removed node IDs.
    """
    if not any(routes):
        return routes, []

    if rng is None:
        rng = Random()

    # 1. Pre-calculate marginal profits
    all_nodes_data = []
    for _, route in enumerate(routes):
        for pos, node in enumerate(route):
            revenue = wastes.get(node, 0.0) * R
            prev = 0 if pos == 0 else route[pos - 1]
            nex = 0 if pos == len(route) - 1 else route[pos + 1]
            detour_cost = float(dist_matrix[prev, node] + dist_matrix[node, nex] - dist_matrix[prev, nex])
            profit = revenue - (detour_cost * C)

            all_nodes_data.append((node, profit))

    if not all_nodes_data:
        return routes, []

    # 2. Sort by profit and pick seed from bottom 25%
    all_nodes_data.sort(key=lambda x: (x[1], x[0]))
    bottom_quartile_size = max(1, len(all_nodes_data) // 4)
    seed_node, seed_profit = all_nodes_data[rng.randint(0, bottom_quartile_size - 1)]

    # 3. Find nodes with similar profit
    candidates = []
    for node, profit in all_nodes_data:
        if node == seed_node:
            continue
        candidates.append((node, abs(profit - seed_profit)))

    candidates.sort(key=lambda x: (x[1], x[0]))
    target_nodes = [x[0] for x in candidates[: n_remove - 1]]
    removed = [seed_node] + target_nodes

    # 4. Efficient route modification
    removed_set: Set[int] = set(removed)
    final_removed = []
    modified_routes = []
    for r in routes:
        new_route = []
        for node in r:
            if node in removed_set:
                final_removed.append(node)
            else:
                new_route.append(node)
        if new_route:
            modified_routes.append(new_route)

    return modified_routes, final_removed
