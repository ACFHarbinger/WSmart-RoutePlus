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
"""

import contextlib
import itertools
from random import Random
from typing import Any, List, Optional, Tuple

import numpy as np


def reversal_delta(dist: Any, segment: List[int], symmetric: Optional[bool] = None) -> float:
    """Cost change of driving ``segment``'s interior arcs in the opposite direction.

    Zero for a symmetric matrix or a segment of fewer than two nodes.

    Args:
        dist: Distance matrix (array-like, indexable by ``[i, j]``) or a
            ``LocalSearch`` with ``d`` (or only ``get_dist``, assumed symmetric).
        segment: Consecutive route nodes, in their current order.
        symmetric: Known symmetry of the matrix; when None it is checked (and
            cached on a ``LocalSearch``).

    Returns:
        float: ``sum(d[b, a]) - sum(d[a, b])`` over the consecutive pairs ``(a, b)``.
    """
    if len(segment) < 2:
        return 0.0
    owner = dist if hasattr(dist, "get_dist") else None
    matrix = getattr(owner, "d", None) if owner is not None else dist
    if matrix is None:
        return 0.0
    try:
        matrix = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return 0.0
    if matrix.ndim != 2:
        return 0.0
    if symmetric is None and owner is not None:
        symmetric = getattr(owner, "_d_symmetric", None)
    if symmetric is None:
        symmetric = bool(np.allclose(matrix, matrix.T))
        if owner is not None:
            with contextlib.suppress(AttributeError):
                owner._d_symmetric = symmetric
    if symmetric:
        return 0.0
    seg = np.asarray(segment)
    return float(matrix[seg[1:], seg[:-1]].sum() - matrix[seg[:-1], seg[1:]].sum())


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


# ---------------------------------------------------------------------------
# Private implementations
# ---------------------------------------------------------------------------


def _apply_2opt(ls, u: int, v: int, r_u: int, p_u: int, p_v: int, exclude_depot: bool = False) -> bool:
    """
    2-opt: reverse the segment between p_u+1 and p_v (inclusive).

    Only applies the move if it reduces total cost.

    Args:
        ls: LocalSearch instance with routes and distance matrix.
        u: First break-point node.
        v: Second break-point node.
        r_u: Route index containing both u and v.
        p_u: Position of u in the route.
        p_v: Position of v in the route.
        exclude_depot: If True, skip moves where p_v is the last position.

    Returns:
        bool: True if the 2-opt reversal was applied; False otherwise.
    """
    if p_u >= p_v:
        return False
    if p_u + 1 == p_v:
        return False

    route = ls.routes[r_u]
    u_next = route[p_u + 1]
    if exclude_depot and p_v == len(route) - 1:
        return False
    v_next = route[p_v + 1] if p_v < len(route) - 1 else 0

    delta = -ls.get_dist(u, u_next) - ls.get_dist(v, v_next) + ls.get_dist(u, v) + ls.get_dist(u_next, v_next)
    delta += reversal_delta(ls, route[p_u + 1 : p_v + 1])

    if delta * ls.C < -1e-4:
        segment = route[p_u + 1 : p_v + 1]
        route[p_u + 1 : p_v + 1] = segment[::-1]
        ls._update_map({r_u})
        return True
    return False


def _apply_3opt(
    ls, u: int, v: int, r_u: int, p_u: int, p_v: int, rng: Random, n_attempts: int = 5, exclude_depot: bool = False
) -> bool:
    """
    3-opt: remove three edges and try all reconnection patterns.

    Randomly selects a third cut point and evaluates four reconnection
    patterns. Applies the best improving move if found.

    Args:
        ls: LocalSearch instance with routes and distance matrix.
        u: First break-point node.
        v: Second break-point node.
        r_u: Route index.
        p_u: Position of u in the route.
        p_v: Position of v in the route.
        rng: Random number generator used to sample the third cut point.
        n_attempts: Number of random third-cut-point attempts.
        exclude_depot: If True, skip moves where the last route position is a cut.

    Returns:
        bool: True if an improving 3-opt move was applied; False otherwise.
    """
    route = ls.routes[r_u]
    if len(route) < 4:
        return False
    if p_u > p_v:
        p_u, p_v = p_v, p_u
        u, v = v, u

    for _ in range(n_attempts):
        p_w = rng.randint(0, len(route) - 1)
        if p_w in {p_u, p_v, p_u + 1, p_v + 1, p_u - 1, p_v - 1}:
            continue

        idx = sorted([p_u, p_v, p_w])
        i, j, k = idx[0], idx[1], idx[2]

        if exclude_depot and k == len(route) - 1:
            continue

        A, B = route[i], route[i + 1] if i < len(route) - 1 else 0
        C, D = route[j], route[j + 1] if j < len(route) - 1 else 0
        E, F = route[k], route[k + 1] if k < len(route) - 1 else 0

        d_base = ls.get_dist(A, B) + ls.get_dist(C, D) + ls.get_dist(E, F)

        g4 = d_base - (ls.get_dist(A, C) + ls.get_dist(B, E) + ls.get_dist(D, F))
        g5 = d_base - (ls.get_dist(A, D) + ls.get_dist(E, B) + ls.get_dist(C, F))
        g6 = d_base - (ls.get_dist(A, D) + ls.get_dist(E, C) + ls.get_dist(B, F))
        g7 = d_base - (ls.get_dist(A, E) + ls.get_dist(D, B) + ls.get_dist(C, F))

        s2, s3 = route[i + 1 : j + 1], route[j + 1 : k + 1]
        r2, r3 = reversal_delta(ls, s2), reversal_delta(ls, s3)
        gains = [g4 - r2 - r3, g5, g6 - r2, g7 - r3]
        best_g = max(gains)
        if best_g * ls.C > 1e-4:
            case = gains.index(best_g)
            s1 = route[: i + 1]
            s2 = route[i + 1 : j + 1]
            s3 = route[j + 1 : k + 1]
            s4 = route[k + 1 :]

            if case == 0:
                route[:] = s1 + s2[::-1] + s3[::-1] + s4
            elif case == 1:
                route[:] = s1 + s3 + s2 + s4
            elif case == 2:
                route[:] = s1 + s3 + s2[::-1] + s4
            elif case == 3:
                route[:] = s1 + s3[::-1] + s2 + s4

            ls._update_map({r_u})
            return True
    return False


def _apply_kopt(
    ls, r_u: int, p_u: int, p_v: int, k: int, rng: Random, n_attempts: int = 5, exclude_depot: bool = False
) -> bool:
    """
    General k-opt for k >= 4.

    Selects *k* cut points by augmenting the two given positions (``p_u``,
    ``p_v``) with ``k − 2`` randomly sampled additional points.  The route
    is sliced into a fixed head, *k* − 1 permutable middle segments, and a
    fixed tail.  Every permutation × orientation of the middle segments is
    evaluated and the best improving configuration is applied.

    Args:
        ls: LocalSearch instance.
        r_u: Route index.
        p_u: First given position.
        p_v: Second given position.
        k: Number of edges to remove (>= 4).
        rng: Random number generator.
        n_attempts: Number of random cut-point sampling attempts.
        exclude_depot: If True, skip configurations where the tail segment
            is empty (depot-adjacent move).

    Returns:
        bool: True if an improving k-opt move was applied; False otherwise.
    """
    route = ls.routes[r_u]
    # A k-opt move on a route needs at least k + 1 nodes
    if len(route) < k + 1:
        return False

    i, j = (p_u, p_v) if p_u < p_v else (p_v, p_u)

    # Multiple random attempts
    for _ in range(n_attempts):
        cuts = _sample_cuts(len(route), i, j, k, rng)
        if cuts is None:
            return False

        head, middle, tail = _get_segments(route, cuts)
        if not middle:
            continue

        if exclude_depot and not tail:
            continue

        original_cost = _connection_cost(ls, head, middle, tail, exclude_depot)
        best_gain, best_config = _find_best_config(ls, head, middle, tail, original_cost, exclude_depot)

        # Apply the best configuration if improving
        if best_config is not None and best_gain * ls.C > 1e-4:
            _apply_config(route, head, middle, tail, best_config)
            ls._update_map({r_u})
            return True

    return False


def _sample_cuts(n: int, p_u: int, p_v: int, k: int, rng: Random) -> Optional[List[int]]:
    """Sample k-2 additional cut points around p_u and p_v.

    Args:
        n: Route length (number of nodes).
        p_u: First fixed cut position.
        p_v: Second fixed cut position.
        k: Total number of cuts desired (>= 2).
        rng: Random number generator.

    Returns:
        Optional[List[int]]: Sorted list of k cut positions, or None if there
        are not enough available positions.
    """
    extra_needed = k - 2
    forbidden = {p_u, p_v}
    for p in (p_u, p_v):
        if p > 0:
            forbidden.add(p - 1)
        if p < n - 1:
            forbidden.add(p + 1)

    available = [p for p in range(n) if p not in forbidden]
    if len(available) < extra_needed:
        return None

    extra = rng.sample(available, extra_needed)
    return sorted([p_u, p_v] + extra)


def _get_segments(route: List[int], cuts: List[int]) -> Tuple[List[int], List[List[int]], List[int]]:
    """Slice the route into head, middle segments, and tail.

    Args:
        route: The full route node sequence.
        cuts: Sorted list of k cut positions (0-indexed).

    Returns:
        Tuple of (head, middle_segments, tail) where head is the fixed prefix,
        tail is the fixed suffix, and middle_segments is the list of permutable
        segments between consecutive cut points.
    """
    segments: List[List[int]] = []
    segments.append(route[: cuts[0] + 1])
    for ci in range(len(cuts) - 1):
        segments.append(route[cuts[ci] + 1 : cuts[ci + 1] + 1])
    segments.append(route[cuts[-1] + 1 :])
    return segments[0], segments[1:-1], segments[-1]


def _find_best_config(
    ls, head: List[int], middle: List[List[int]], tail: List[int], original_cost: float, exclude_depot: bool = False
) -> Tuple[float, Optional[Tuple[Tuple[int, ...], Tuple[bool, ...]]]]:
    """Enumerate all permutations and orientations to find best gain.

    Args:
        ls: LocalSearch instance with distance matrix.
        head: Fixed head segment.
        middle: List of permutable middle segments.
        tail: Fixed tail segment.
        original_cost: Inter-segment edge cost of the current configuration.
        exclude_depot: Passed through to :func:`_connection_cost`.

    Returns:
        Tuple of (best_gain, best_config) where best_config is
        (permutation_tuple, reversal_tuple), or (0.0, None) if no improvement.
    """
    best_gain = 0.0
    best_config = None
    n_middle = len(middle)
    identity_perm = tuple(range(n_middle))
    for perm in itertools.permutations(range(n_middle)):
        for orient_bits in range(1 << n_middle):
            reversals = tuple(bool(orient_bits & (1 << b)) for b in range(n_middle))
            if perm == identity_perm and not any(reversals):
                continue

            reordered = []
            for idx, seg_idx in enumerate(perm):
                seg = middle[seg_idx]
                reordered.append(seg[::-1] if reversals[idx] else seg)

            new_cost = _connection_cost(ls, head, reordered, tail, exclude_depot)
            new_cost += sum(reversal_delta(ls, middle[s]) for s, r in zip(perm, reversals, strict=True) if r)
            gain = original_cost - new_cost

            if gain > best_gain:
                best_gain = gain
                best_config = (perm, reversals)
    return best_gain, best_config


def _apply_config(
    route: List[int],
    head: List[int],
    middle: List[List[int]],
    tail: List[int],
    config: Tuple[Tuple[int, ...], Tuple[bool, ...]],
) -> None:
    """Apply the selected configuration to the route.

    Args:
        route: The route list to overwrite in-place.
        head: Fixed head segment.
        middle: List of middle segments in their original order.
        tail: Fixed tail segment.
        config: Tuple of (permutation, reversals) produced by
            :func:`_find_best_config`.
    """
    perm, reversals = config
    new_route: List[int] = list(head)
    for idx, seg_idx in enumerate(perm):
        seg = middle[seg_idx]
        if reversals[idx]:
            new_route.extend(seg[::-1])
        else:
            new_route.extend(seg)
    new_route.extend(tail)
    route[:] = new_route


def _connection_cost(
    ls: Any, head: List[int], middle: List[List[int]], tail: List[int], exclude_depot: bool = False
) -> float:
    """
    Compute the total cost of edges connecting consecutive segments.

    Only counts the *k* edges that a k-opt move breaks/reconnects; the interior
    cost of a reversed segment is added by the caller (:func:`reversal_delta`).

    Args:
        ls: LocalSearch instance.
        head: Head segment (fixed).
        middle: List of middle segments in their current order.
        tail: Tail segment (fixed).
        exclude_depot: If True, skip cost terms where a segment boundary
            involves an empty segment (depot sentinel).

    Returns:
        float: Sum of inter-segment edge costs.
    """
    cost = 0.0
    all_segs = [head] + middle + [tail]

    for i in range(len(all_segs) - 1):
        seg_a = all_segs[i]
        seg_b = all_segs[i + 1]
        # Last node of seg_a -> first node of seg_b
        if exclude_depot and (not seg_a or not seg_b):
            continue

        node_a = seg_a[-1] if seg_a else 0
        node_b = seg_b[0] if seg_b else 0
        cost += ls.get_dist(node_a, node_b)

    return cost
