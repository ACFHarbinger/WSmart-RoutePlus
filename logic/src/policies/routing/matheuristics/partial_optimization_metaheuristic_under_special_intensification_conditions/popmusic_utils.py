"""Assembly helpers for the POPMUSIC matheuristic.

Holds the route-centroid computation, the KDTree-based nearest-route lookup
and the O(r) adjacency-list subproblem assembly, split out of ``solver``.

Issue 2 — KDTree inefficiency:
    The proximity network G is built once during initialization as a static
    adjacency list dict[int, list[int]] sorted by inter-center distance. Subproblem
    assembly walks G in O(r). Centroids are updated in O(r) after each improvement.
    G topology is never rebuilt — consistent with the paper's proximate optimality
    assumption.
"""

from __future__ import annotations

from collections import deque
from typing import Any, List, Set

import numpy as np

from .state import POPMUSICState


def _route_centroid(
    nodes: List[int],
    coord_array: np.ndarray,
) -> np.ndarray:
    """
    Compute the geographic centroid (mean coordinate) of non-depot nodes.

    Args:
        nodes: Route node list including depot sentinels (0).
        coord_array: Shape (n+1, 2) — all node coordinates.

    Returns:
        centroid: Shape (2,) float64.
    """
    customers = [n for n in nodes if n != 0]
    if not customers:
        return coord_array[0].copy()
    return coord_array[customers].mean(axis=0)


def find_route_neighbors(
    seed_idx: int,
    centroids: List[np.ndarray],
    k: int = 5,
    kdtree: Any = None,
    k_prox: int = 10,
) -> List[int]:
    """
    Find the k nearest route neighbors of a seed route using a KDTree.

    Queries the KDTree for k_prox candidates around the seed centroid, then
    returns the k nearest indices (including the seed itself at position 0,
    since KDTree includes the query point).

    Args:
        seed_idx: Index of the seed route in centroids.
        centroids: List of route centroid coordinate arrays.
        k: Number of neighbors to return (including the seed itself).
        kdtree: Pre-built scipy.spatial.KDTree over the centroid coordinates.
        k_prox: Number of candidates to fetch from the KDTree before selecting k.

    Returns:
        List of route indices (length <= k), sorted by ascending distance.
    """
    n_query = min(k_prox, len(centroids))
    _, indices = kdtree.query(np.asarray(centroids[seed_idx]), k=n_query)
    if np.ndim(indices) == 0:
        return [int(indices)]
    return list(indices[:k])


def _assemble_subproblem(
    seed_slot: int,
    state: POPMUSICState,
    r: int,
) -> List[int]:
    """
    Assemble a subproblem of at most r parts by walking the adjacency list G.

    Time complexity: O(r) — independent of n (Issue 2 guarantee).

    The walk is a BFS/greedy expansion from seed_slot: at each step, the
    unvisited neighbor with the smallest d_prox (pre-sorted in G) is added
    until r parts are collected or G is exhausted.

    Args:
        seed_slot: Slot index of the seed part.
        state: Current POPMUSICState.
        r: Subproblem size (number of parts).

    Returns:
        List of slot indices forming the subproblem (length ≤ r).
    """
    selected: List[int] = [seed_slot]
    visited: Set[int] = {seed_slot}
    frontier: deque[int] = deque([seed_slot])

    while frontier and len(selected) < r:
        current = frontier.popleft()
        for neighbor in state.G.get(current, []):
            if neighbor not in visited and neighbor in state.active:
                visited.add(neighbor)
                selected.append(neighbor)
                frontier.append(neighbor)
                if len(selected) >= r:
                    break

    return selected
