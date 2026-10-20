"""Sequential search and move evaluation routines for the Lin-Kernighan heuristic."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.helpers.operators.search_heuristics._objective import (
    get_cost,
    is_better,
)


def _build_pos_map(tour: List[int]) -> np.ndarray:
    """
    Build an O(1) position-lookup array from a closed tour.

    Args:
        tour: Closed tour of length n+1 (tour[0] == tour[-1]).

    Returns:
        Integer array of shape (max_node+1,) where pos[v] = index of v in
        the open prefix tour[0..n-1].
    """
    n = len(tour) - 1
    max_node = max(tour) + 1
    pos = np.empty(max_node, dtype=np.int32)
    for i in range(n):
        pos[tour[i]] = i
    return pos


def _apply_2opt_positions(
    tour: List[int],
    pos: np.ndarray,
    p1: int,
    p2: int,
) -> Tuple[List[int], np.ndarray]:
    """
    Apply a 2-opt reversal between positions p1 and p2 (inclusive).

    Reverses the segment tour[p1..p2] in-place and updates the position map.

    Args:
        tour: Open node list (length n); modified in-place.
        pos:  Position array; updated in-place.
        p1:   Start of reversal segment (inclusive).
        p2:   End of reversal segment (inclusive); p2 > p1.

    Returns:
        The same (mutated) tour and pos objects.
    """
    while p1 < p2:
        tour[p1], tour[p2] = tour[p2], tour[p1]
        pos[tour[p1]] = p1
        pos[tour[p2]] = p2
        p1 += 1
        p2 -= 1
    return tour, pos


def _succ(p: int, n: int) -> int:
    """Position of the successor (wraps around).

    Args:
        p: Current position.
        n: Total number of nodes.

    Returns:
        int: Successor position.
    """
    return (p + 1) % n


def _pred(p: int, n: int) -> int:
    """Position of the predecessor (wraps around).

    Args:
        p: Current position.
        n: Total number of nodes.

    Returns:
        int: Predecessor position.
    """
    return (p - 1) % n


def _lk_search(  # noqa: C901
    t1: int,
    tour: List[int],
    pos: np.ndarray,
    n: int,
    d: np.ndarray,
    candidates: Dict[int, List[int]],
    max_depth: int,
) -> Optional[Tuple[List[int], np.ndarray, float]]:
    """
    Run the sequential Lin-Kernighan search starting at node t1.

    Implements Algorithm 1 of Lin & Kernighan (1973) with:
    - Positive gain criterion at every level.
    - Sequential constraint: t_{2i} is the tour-neighbour of t_{2i-1} that
      maintains the open-path structure (both directions tried).
    - Backtracking: all valid (t3, t4) combinations at each depth are
      enumerated before giving up.
    - First-improvement acceptance: returns immediately upon finding a
      strictly improving tour.

    The function works on a *copy* of the current tour and position array
    so that failed attempts do not corrupt the caller's state.

    Args:
        t1:         Root node for this search.
        tour:       Open tour list (length n, no closing duplicate).
        pos:        Position map (index by node → position in tour).
        n:          Number of nodes.
        d:          Distance matrix.
        candidates: α-nearest-neighbour candidate lists.
        max_depth:  Maximum sequential search depth (2..5).

    Returns:
        Optional[Tuple[List[int], np.ndarray, float]]: (new_open_tour, new_pos, new_cost)
        if an improving move was found; ``None`` otherwise.
    """
    p1 = int(pos[t1])
    # Try both tour-adjacencies of t1 as the first removed edge
    for direction in (_succ(p1, n), _pred(p1, n)):
        p2 = direction
        t2 = tour[p2]
        g1_base = d[t1, t2]  # partial gain after removing (t1, t2)
        if g1_base < 1e-9:
            continue

        # Backtracking stack:
        # Each frame: (depth, free_end_node, free_end_pos, G_so_far,
        #              tour_copy, pos_copy, break_positions)
        # We use an explicit stack instead of recursion to avoid Python
        # recursion-depth issues on large instances.
        initial_tour = list(tour)
        initial_pos = pos.copy()

        # Level 1 candidates: α-neighbours of t2
        for t3 in candidates[t2]:
            if t3 == t1:
                continue
            c_added = d[t2, t3]
            G1 = g1_base - c_added  # G after adding (t2, t3)
            if G1 <= 1e-9:
                # Candidates are sorted by α; remaining ones will only be worse
                break

            p3 = int(initial_pos[t3])
            # Sequential constraint: t4 is the tour-neighbour of t3 that is
            # "between" t3 and t1 (try both, but skip t2 and t1 itself)
            for p4 in (_succ(p3, n), _pred(p3, n)):
                t4 = initial_tour[p4]
                if t4 in (t2, t1):
                    continue
                g_removed = d[t3, t4]  # gain of removing (t3, t4)
                G2 = G1 + g_removed

                # ---- Try closing at depth 2 (2-opt move) ----
                closing_cost = d[t4, t1]
                if G2 - closing_cost > 1e-9:
                    # Valid improving 2-opt: reverse segment between p2 and p3
                    # (or between p4 and p1, depending on orientation)
                    w_tour = list(initial_tour)
                    w_pos = initial_pos.copy()

                    # The 2-opt reversal: between positions p2 and p3
                    lo, hi = (min(p2, p3), max(p2, p3))
                    _apply_2opt_positions(w_tour, w_pos, lo, hi)

                    closed = w_tour + [w_tour[0]]
                    new_cost = get_cost(closed, d)
                    ref_cost = get_cost(initial_tour + [initial_tour[0]], d)
                    if is_better(new_cost, ref_cost):
                        return w_tour, w_pos, new_cost

                # ---- Try extending to depth 3..max_depth ----
                if max_depth >= 3:
                    result = _lk_extend(
                        t1=t1,
                        t_free=t4,
                        p_free=p4,
                        G=G2,
                        depth=3,
                        tour=list(initial_tour),
                        pos=initial_pos.copy(),
                        n=n,
                        d=d,
                        candidates=candidates,
                        max_depth=max_depth,
                        broken={(min(t1, t2), max(t1, t2)), (min(t3, t4), max(t3, t4))},
                        added={(min(t2, t3), max(t2, t3))},
                    )
                    if result is not None:
                        return result

    return None


def _lk_extend(  # noqa: C901
    t1: int,
    t_free: int,
    p_free: int,
    G: float,
    depth: int,
    tour: List[int],
    pos: np.ndarray,
    n: int,
    d: np.ndarray,
    candidates: Dict[int, List[int]],
    max_depth: int,
    broken: set,
    added: set,
) -> Optional[Tuple[List[int], np.ndarray, float]]:
    """
    Recursive extension of the sequential LK search.

    Called at depth ≥ 3.  At each level:
    1. Iterate over α-candidates of t_free as the next added-edge endpoint t_next.
    2. Apply the gain criterion: G - d[t_free, t_next] > 0.
    3. Try closing: is G - d[t_free, t_next] + d[t_next, t4_next] - d[t4_next, t1] > 0?
    4. If not improving and depth < max_depth, recurse.

    Args:
        t1:        Root node (constant across all recursive calls).
        t_free:    Current "free" endpoint (last node removed via t_{2i}).
        p_free:    Position of t_free in the open tour.
        G:         Accumulated gain up to this point.
        depth:     Current search depth (number of added edges + 1).
        tour:      Mutable open tour list for this branch.
        pos:       Mutable position map for this branch.
        n:         Number of nodes.
        d:         Distance matrix.
        candidates: α-candidate lists.
        max_depth:  Search depth limit.
        broken:    Set of removed edges (as (min, max) node pairs).
        added:     Set of added edges (as (min, max) node pairs).

    Returns:
        Optional[Tuple[List[int], np.ndarray, float]]: (new_tour, new_pos, new_cost)
        if an improving tour was found; else None.
    """
    ref_cost = get_cost(tour + [tour[0]], d)

    for t_next in candidates[t_free]:
        if t_next == t1 and depth < 3:
            continue
        edge_add = (min(t_free, t_next), max(t_free, t_next))
        if edge_add in added or edge_add in broken:
            continue

        c_add = d[t_free, t_next]
        G_new = G - c_add
        if G_new <= 1e-9:
            break  # sorted candidates → remaining are worse

        p_next = int(pos[t_next])

        # Try both neighbours of t_next as the next removed edge
        for p4_next in (_succ(p_next, n), _pred(p_next, n)):
            t4_next = tour[p4_next]
            if t4_next == t_free:
                continue
            edge_rem = (min(t_next, t4_next), max(t_next, t4_next))
            if edge_rem in added or edge_rem in broken:
                continue

            g_rem = d[t_next, t4_next]
            G_ext = G_new + g_rem

            # Closing attempt: add (t4_next, t1)
            closing = d[t4_next, t1]
            if G_ext - closing > 1e-9:
                # Attempt to apply the multi-opt move by direct tour reconstruction
                # For depth ≥ 3 we perform a simplified 3-opt-style reconstruction:
                # reverse the segment between p_free and p_next (the "added" segment)
                # This handles the sequential 3-opt case.
                w_tour = list(tour)
                w_pos = pos.copy()
                lo, hi = (min(p_free, p_next), max(p_free, p_next))
                _apply_2opt_positions(w_tour, w_pos, lo, hi)
                closed = w_tour + [w_tour[0]]
                new_cost = get_cost(closed, d)
                if is_better(new_cost, ref_cost):
                    return w_tour, w_pos, new_cost

            # Recurse if budget allows
            if depth < max_depth:
                result = _lk_extend(
                    t1=t1,
                    t_free=t4_next,
                    p_free=p4_next,
                    G=G_ext,
                    depth=depth + 1,
                    tour=list(tour),
                    pos=pos.copy(),
                    n=n,
                    d=d,
                    candidates=candidates,
                    max_depth=max_depth,
                    broken=broken | {edge_rem},
                    added=added | {edge_add},
                )
                if result is not None:
                    return result

    return None
