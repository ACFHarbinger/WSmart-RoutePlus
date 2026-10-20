"""
k-opt Intra-Route Operator Module.

This module implements the generalized k-opt intra-route operator, which removes
k edges from a route and reconnects the segments optimally to reduce tour length.

Provides the unified ``move_kopt_intra`` function and convenience wrappers:

- ``move_2opt_intra`` → delegates to ``move_kopt_intra(k=2)``
- ``move_3opt_intra`` → delegates to ``move_kopt_intra(k=3, rng=rng)``

Higher values of *k* systematically enumerate all permutations and
orientations of the middle segments and apply the best improving move.

Every move is priced with the directed distances: reversing a segment also
changes the cost of its interior arcs when the matrix is asymmetric
(:func:`reversal_delta`), which a symmetric delta of the exchanged edges misses.

Attributes:
    reversal_delta: Cost change of driving a segment in the opposite direction.
    directed_two_opt_route: Best-improvement 2-opt with exact directed costs.

Example:
    >>> from logic.src.policies.helpers.operators.intra_route.k_opt import move_kopt_intra
    >>> improved = move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=2)
    >>> improved = move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=3, rng=rng)
    >>> improved = move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=4, rng=rng)
    >>> improved_route = two_opt_route(route, dist_matrix)

The in-place move implementations (``_apply_2opt``, ``_apply_3opt``, ``_apply_kopt`` and their
helpers) and :func:`reversal_delta` live in ``k_opt_apply.py``; this module re-exports
``reversal_delta``.
"""

import itertools
from random import Random
from typing import List, Optional

import numpy as np

from .k_opt_apply import _apply_2opt, _apply_3opt, _apply_kopt
from .k_opt_apply import reversal_delta as reversal_delta


def directed_two_opt_route(route: List[int], dist_matrix: np.ndarray, max_moves: int = 10_000) -> List[int]:
    """Best-improvement 2-opt of one route with exact directed move costs.

    Every segment reversal, including those touching the depot, is priced with
    the arcs it removes, the arcs it adds and its interior driven backwards.

    Args:
        route: Ordered bins, depot excluded.
        dist_matrix: Directed distance matrix (index 0 = depot).
        max_moves: Cap on applied reversals.

    Returns:
        List[int]: The improved route, depot excluded.
    """
    full = np.array([0, *route, 0])
    n = len(full)
    if n < 4:
        return list(route)
    d = np.asarray(dist_matrix, dtype=float)
    i_idx, j_idx = np.triu_indices(n - 1, 1)
    keep = (i_idx >= 1) & (j_idx <= n - 2)
    i_idx, j_idx = i_idx[keep], j_idx[keep]
    for _ in range(max_moves):
        fwd, bwd = d[full[:-1], full[1:]], d[full[1:], full[:-1]]
        pf = np.concatenate([[0.0], np.cumsum(fwd)])
        pb = np.concatenate([[0.0], np.cumsum(bwd)])
        # Reversing full[i..j]: arcs (i-1, i) and (j, j+1) change, the interior runs backwards.
        old = fwd[i_idx - 1] + fwd[j_idx] + (pf[j_idx] - pf[i_idx])
        new = d[full[i_idx - 1], full[j_idx]] + d[full[i_idx], full[j_idx + 1]] + (pb[j_idx] - pb[i_idx])
        delta = new - old
        k = int(np.argmin(delta))
        if delta[k] >= -1e-9:
            break
        i, j = i_idx[k], j_idx[k]
        full[i : j + 1] = full[i : j + 1][::-1].copy()
    return [int(x) for x in full[1:-1]]


def two_opt_route(
    route: List[int],
    dist_matrix: np.ndarray,
    max_iter: Optional[int] = 200,
    exclude_depot: bool = False,
    tol: float = 1e-9,
) -> List[int]:
    """
    Apply 2-opt to a single route (iterative, first-improvement).

    Args:
        route:         Ordered node list, depot excluded.
        dist_matrix:   Full distance matrix (index 0 = depot).
        max_iter:      Maximum improvement passes; None runs until no reversal improves.
        exclude_depot: If True, skips edges connected to node 0.
        tol:           A reversal is applied when it shortens the route by more than ``tol``.

    Returns:
        Improved route (depot excluded).
    """
    if len(route) < 2:
        return route

    route = list(route)
    full = [0] + route + [0]
    n = len(full)
    symmetric = bool(np.allclose(dist_matrix, np.transpose(dist_matrix)))

    for _ in itertools.count() if max_iter is None else range(max_iter):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                if exclude_depot and (i == 1 or j == n - 2):
                    continue
                # Edges: full[i-1]->full[i] and full[j]->full[j+1]
                delta = (
                    dist_matrix[full[i - 1], full[j]]
                    + dist_matrix[full[i], full[j + 1]]
                    - dist_matrix[full[i - 1], full[i]]
                    - dist_matrix[full[j], full[j + 1]]
                    + reversal_delta(dist_matrix, full[i : j + 1], symmetric)
                )
                if delta < -tol:
                    full[i : j + 1] = full[i : j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return full[1:-1]


def three_opt_route(
    route: List[int],
    dist_matrix: np.ndarray,
    max_iter: int = 50,
    exclude_depot: bool = False,
) -> List[int]:
    """
    Apply 3-opt to a single route (iterative, first-improvement).

    Args:
        route:         Ordered node list, depot excluded.
        dist_matrix:   Full distance matrix (index 0 = depot).
        max_iter:      Maximum improvement passes.
        exclude_depot: If True, skips edges connected to node 0.

    Returns:
        Improved route (depot excluded).
    """
    if len(route) < 4:
        return two_opt_route(route, dist_matrix, max_iter, exclude_depot)

    def _route_cost(r: List[int]) -> float:
        f = [0] + r + [0]
        cost = 0.0
        for i in range(len(f) - 1):
            if exclude_depot and (i == 0 or i == len(f) - 2):
                continue
            cost += dist_matrix[f[i], f[i + 1]]
        return cost

    for _ in range(max_iter):
        improved = False
        n = len(route)
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    A = route[: i + 1]
                    B = route[i + 1 : j + 1]
                    C = route[j + 1 : k + 1]
                    D = route[k + 1 :]

                    curr_cost = _route_cost(route)
                    candidates = [
                        A + B[::-1] + C + D,
                        A + B + C[::-1] + D,
                        A + C + B + D,
                        A + C + B[::-1] + D,
                    ]
                    for cand in candidates:
                        if _route_cost(cand) < curr_cost - 1e-9:
                            route = cand
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return route


def move_2opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    2-opt intra-route operator: reverse a segment within a route.

    Convenience wrapper for ``move_kopt_intra(k=2)``.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: Start node of the segment (edge u->next_u broken).
        v: End node of the segment (edge v->next_v broken).
        r_u: Index of the route (must equal r_v for intra-route).
        p_u: Position of u in the route.
        r_v: Index of the route (unused but required for signature).
        p_v: Position of v in the route.

    Returns:
        bool: True if the reversal was applied (improving), False otherwise.
    """
    return move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=2)


def move_3opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: Random) -> bool:
    """
    3-opt intra-route operator: reconnect three segments within a route.

    Convenience wrapper for ``move_kopt_intra(k=3, rng=rng)``.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: First cut point node.
        v: Second cut point node.
        r_u: Index of the route.
        p_u: Position of u in the route.
        r_v: Index of the route (unused but required for signature).
        p_v: Position of v in the route.
        rng: Random number generator.

    Returns:
        bool: True if a 3-opt move was applied (improving), False otherwise.
    """
    return move_kopt_intra(ls, u, v, r_u, p_u, r_v, p_v, k=3, rng=rng)


def move_kopt_intra(
    ls,
    u: int,
    v: int,
    r_u: int,
    p_u: int,
    r_v: int,
    p_v: int,
    k: int = 2,
    rng: Optional[Random] = None,
    exclude_depot: bool = False,
) -> bool:
    """
    Generalized k-opt intra-route operator.

    Removes *k* edges from a single route and reconnects the resulting
    segments in the best improving configuration found. Subsumes 2-opt
    (segment reversal), 3-opt (triple-edge reconnection), and higher-order
    moves.

    For *k* cut points the route is split into *k* + 1 segments. The head
    and tail segments stay fixed; the *k* − 1 middle segments are tried in
    every permutation and orientation. The configuration with the largest
    cost saving is applied.

    Complexity per call:
        - k=2: O(1) — single delta check.
        - k=3: O(1) — 4 reconnection patterns plus up to 5 random samples.
        - k≥4: O((k−1)! · 2^(k−1)) reconnection patterns per sample.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: First breakpoint node.
        v: Second breakpoint node.
        r_u: Index of the route containing u.
        p_u: Position of u in the route.
        r_v: Index of the route containing v (must equal r_u for intra-route).
        p_v: Position of v in the route.
        k: Number of edges to remove (≥ 2). Defaults to 2.
        rng: Random number generator, required when ``k >= 3``.
        exclude_depot: If True, moves that would place the depot-adjacent
            edges in the middle of a segment are rejected.

    Returns:
        bool: True if an improving move was applied, False otherwise.

    Raises:
        ValueError: If *k* < 2.
        ValueError: If *k* >= 3 and *rng* is not provided.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")

    if k == 2:
        return _apply_2opt(ls, u, v, r_u, p_u, p_v, exclude_depot)

    if rng is None:
        raise ValueError("rng is required for k >= 3")

    if k == 3:
        return _apply_3opt(ls, u, v, r_u, p_u, p_v, rng, exclude_depot=exclude_depot)

    return _apply_kopt(ls, r_u, p_u, p_v, k, rng, exclude_depot=exclude_depot)
