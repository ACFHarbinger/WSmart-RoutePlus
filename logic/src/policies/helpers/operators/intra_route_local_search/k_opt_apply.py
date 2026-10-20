"""In-place k-opt move implementations on a local-search state.

Holds :func:`reversal_delta` (the directed cost change of reversing a segment) and the
private 2-opt, 3-opt and general k-opt move implementations that
:func:`~logic.src.policies.helpers.operators.intra_route_local_search.k_opt.move_kopt_intra`
dispatches to, with their cut sampling, segment and reconnection-cost helpers. Moved
unchanged from ``k_opt.py``, which re-exports :func:`reversal_delta`.
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
