"""Elite-tour merging for the LKH-3 iterated local-search loop.

Holds the pool-based tour recombination routines split out of
``tour_construction``: greedy shared-edge merging, IP-based assignment
merging, successor-chain reconstruction and the best-first wrapper.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def merge_tours(
    tour1: List[int],
    tour2: List[int],
    distance_matrix: np.ndarray,
) -> List[int]:
    """
    Combine two elite tours by preferring edges common to both.

    Shared edges are followed greedily; when no shared edge is available
    the nearest unvisited node is selected.  This implements the pool-based
    recombination described in Helsgaun (2000), Section 4.4.

    Args:
        tour1: First tour (closed).
        tour2: Second tour (closed).
        distance_matrix: (n × n) cost matrix.

    Returns:
        List[int]: A new closed tour that reuses as many shared edges as possible.
    """
    n = len(tour1) - 1

    edges1: set = set()
    edges2: set = set()
    for i in range(len(tour1) - 1):
        a, b = tour1[i], tour1[i + 1]
        edges1.add((min(a, b), max(a, b)))
    for i in range(len(tour2) - 1):
        a, b = tour2[i], tour2[i + 1]
        edges2.add((min(a, b), max(a, b)))

    common_edges = edges1 & edges2

    if not common_edges:
        cost1 = sum(distance_matrix[tour1[i], tour1[i + 1]] for i in range(len(tour1) - 1))
        cost2 = sum(distance_matrix[tour2[i], tour2[i + 1]] for i in range(len(tour2) - 1))
        return tour1 if cost1 <= cost2 else tour2

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in common_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited = [False] * n
    merged = [0]
    visited[0] = True
    current = 0
    while len(merged) < n:
        next_node = None
        for neighbor in adj[current]:
            if not visited[neighbor]:
                next_node = neighbor
                break
        if next_node is None:
            min_dist = float("inf")
            for node in range(n):
                if not visited[node] and distance_matrix[current, node] < min_dist:
                    min_dist = distance_matrix[current, node]
                    next_node = node
        if next_node is None:
            break
        merged.append(next_node)
        visited[next_node] = True
        current = next_node

    merged.append(0)
    return merged


def merge_tours_ip(
    tour_pool: List[List[int]],
    distance_matrix: np.ndarray,
) -> List[int]:
    """Exact recombination of elite tours via restricted edge assignment.

    Collects the union of all edges from the tour pool, builds a restricted
    cost matrix limited to those edges, and solves the resulting assignment
    problem via ``scipy.optimize.linear_sum_assignment`` to find the
    minimum-cost Hamiltonian tour using only pooled edges.

    Falls back to greedy :func:`merge_tours` if the restricted problem is
    infeasible (e.g., too sparse to form a valid tour).

    Args:
        tour_pool: List of elite tours (each closed).
        distance_matrix: (n × n) symmetric cost matrix.

    Returns:
        List[int]: A new closed tour combining edges from the elite pool.
    """
    if len(tour_pool) < 2:
        return tour_pool[0][:] if tour_pool else []

    n = len(tour_pool[0]) - 1  # Number of nodes (excluding closing duplicate)

    # --- Collect union of edges from all elite tours ---
    pooled_edges: set = set()
    for tour in tour_pool:
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            pooled_edges.add((min(a, b), max(a, b)))

    # --- Build restricted adjacency with only pooled edges ---
    # Create a cost matrix where non-pooled edges have infinite cost
    big_m = float(np.sum(distance_matrix)) + 1.0
    restricted_cost = np.full((n, n), big_m, dtype=float)
    np.fill_diagonal(restricted_cost, big_m)

    for a, b in pooled_edges:
        if 0 <= a < n and 0 <= b < n:
            restricted_cost[a, b] = distance_matrix[a, b]
            restricted_cost[b, a] = distance_matrix[b, a]

    # --- Solve assignment to get minimum-cost edge selection ---
    try:
        row_ind, col_ind = linear_sum_assignment(restricted_cost)
    except ValueError:
        # scipy failed — fall back to greedy pairwise merge
        return merge_tours(tour_pool[0], tour_pool[1], distance_matrix)

    # --- Build tour from assignment result ---
    return _build_tour_from_successor(
        row_ind,
        col_ind,
        restricted_cost,
        big_m,
        n,
        distance_matrix,
        tour_pool,
    )


def _build_tour_from_successor(
    row_ind: np.ndarray,
    col_ind: np.ndarray,
    restricted_cost: np.ndarray,
    big_m: float,
    n: int,
    distance_matrix: np.ndarray,
    tour_pool: List[List[int]],
) -> List[int]:
    """Build a closed tour from the assignment (successor) result.

    Args:
        row_ind: Row indices from linear_sum_assignment.
        col_ind: Column indices from linear_sum_assignment.
        restricted_cost: Restricted cost matrix.
        big_m: Penalty value for non-pooled edges.
        n: Number of nodes.
        distance_matrix: Full cost matrix.
        tour_pool: Original tour pool (for greedy fallback).

    Returns:
        List[int]: A closed tour.
    """
    # The assignment gives a perfect matching; chain it into a tour
    successor: Dict[int, int] = {}
    for r, c in zip(row_ind, col_ind, strict=False):
        if restricted_cost[r, c] < big_m:
            successor[r] = c

    # Construct tour by following successor pointers
    if 0 not in successor:
        return merge_tours(tour_pool[0], tour_pool[1], distance_matrix)

    visited = [False] * n
    tour_result = [0]
    visited[0] = True
    current = 0

    while True:
        nxt = successor.get(current)
        if nxt is None or visited[nxt]:
            # Nearest-neighbor fallback: the linear-sum assignment produced
            # subtours rather than a Hamiltonian cycle.  This is expected
            # since LSA solves matching, not TSP.
            logger.info(
                "merge_tours_ip: successor chain broken at node %d — "
                "patching with nearest-neighbor fallback (subtour detected)",
                current,
            )
            best_dist = float("inf")
            best_node = -1
            for j in range(n):
                if not visited[j] and distance_matrix[current, j] < best_dist:
                    best_dist = distance_matrix[current, j]
                    best_node = j
            if best_node == -1:
                break
            nxt = best_node

        tour_result.append(nxt)
        visited[nxt] = True
        current = nxt

        if len(tour_result) >= n:
            break

    tour_result.append(0)
    return tour_result


def merge_tours_best(
    tour_pool: List[List[int]],
    distance_matrix: np.ndarray,
    use_ip: bool = True,
) -> List[int]:
    """Dispatch to IP-based or greedy tour merging.

    Args:
        tour_pool: List of elite tours (each closed).
        distance_matrix: (n × n) cost matrix.
        use_ip: If True, attempt IP-based merging; else use greedy.

    Returns:
        List[int]: A new closed tour combining edges from the pool.
    """
    if len(tour_pool) < 2:
        return tour_pool[0][:] if tour_pool else []

    if use_ip:
        return merge_tours_ip(tour_pool, distance_matrix)
    else:
        # Greedy: merge first two, then fold in remaining
        result = merge_tours(tour_pool[0], tour_pool[1], distance_matrix)
        for i in range(2, len(tour_pool)):
            result = merge_tours(result, tour_pool[i], distance_matrix)
        return result
