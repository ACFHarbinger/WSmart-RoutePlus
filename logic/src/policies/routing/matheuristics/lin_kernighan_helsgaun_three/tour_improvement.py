"""
LKH-3 Tour Improvement Components.

Implements the sequential k-opt local-search loop (k = 2..5) used
in the Lin-Kernighan-Helsgaun heuristic (Helsgaun 2000, Sections 3–4).

These routines iterate over potential cut positions and use exact gain
pre-screening before delegating the actual segment swaps to the shared
:func:`move_kopt_intra` operator.

The high-level driver for these components is :func:`_improve_tour` in
``lkh3_improvement``.

The 2-/3-opt routines and the shared acceptance gate moved to
:mod:`kopt_moves_2_3`, the 4-/5-opt routines to :mod:`kopt_moves_4_5` and
the Or-opt routines to :mod:`or_opt_moves`; all of them are re-imported here
so every existing ``from tour_improvement import ...`` keeps working. This
module keeps the dynamic topology-discovery search and the move
verification/reconstruction helper.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.load_tracker import (
    LoadState,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.objective import (
    get_score,
    is_better,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_construction import (
    build_tour_from_segments,
)


def _dynamic_kopt_search(
    curr_tour: List[int],
    i: int,
    t1: int,
    t2: int,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    max_k: int = 5,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Dynamic, recursive k-opt search following true sequential LKH-3 principles.

    Refined Logic:
    1. Positive-G Pruning: Prunes if G_partial - cost(added_edge) <= 0.
    2. Sequential branching: t_{2k+2} is the unique non-backtracking neighbor of t_{2k+1}.
    3. Efficiency: Uses the O(1) pos lookup array passed from the driver.

    Args:
        curr_tour: The current tour.
        i: The index of the first cut position.
        t1: The first node.
        t2: The second node.
        candidates: Candidates.
        distance_matrix: Distance matrix.
        waste: Waste.
        capacity: Capacity.
        rng: Random number generator.
        n_original: Number of original nodes.
        load_state: Load state.
        max_k: Maximum k value.
        pos: Position of nodes.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improving move found.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # pos array is required for O(1) lookups
    if pos is None:
        pos = np.full(len(distance_matrix), -1, dtype=np.int32)
        for idx, node in enumerate(curr_tour[:-1]):
            pos[node] = idx

    def backtrack(
        k: int,
        t_list: List[int],
        gain_c: float,
    ) -> Tuple[Optional[List[int]], float, float, bool]:
        """
        Recursive core of the sequential LK search.
        k: current level (from 1 to max_k)
        t_list: points [t1, t2, ..., t_{2k}]
        gain_c: cumulative distance gain
        """
        if k > max_k:
            return None, 0.0, 0.0, False

        t_prev = t_list[-1]  # t_{2k}

        # Select t_{2k+1} from candidate list of t_{2k}
        for t_next in candidates[t_prev]:
            if t_next in t_list:
                continue

            # Step 1: Add edge (t_{2k}, t_{2k+1})
            added_c = d[t_prev, t_next]

            # Step 2: Positive-G Criterion (Prune if extension cannot help)
            # True LKH prunes when gain_c - added_c <= 0
            G_after_add = gain_c - added_c
            if G_after_add <= 0:
                continue

            # Step 3: Select t_{2k+2} - must keep the move sequential and non-backtracking
            idx_next = int(pos[t_next])  # type: ignore[index]
            pred = curr_tour[(idx_next - 1) % nodes_count]
            succ = curr_tour[(idx_next + 1) % nodes_count]
            t_neighbor = succ if pred == t_prev else pred

            if t_neighbor in t_list:
                continue

            # Step 4: Break edge (t_{2k+1}, t_{2k+2})
            broken_c = d[t_next, t_neighbor]
            new_gain_c = G_after_add + broken_c

            # Step 5: Try closure with (t_{2k+2}, t1)
            final_added_c = d[t_neighbor, t1]
            final_delta_c = new_gain_c - final_added_c

            if final_delta_c > 1e-6:
                # Validate closure and check penalty/score
                t_all = t_list + [t_next, t_neighbor]
                new_t, p_new, c_new, improved = _verify_and_construct(
                    curr_tour, t_all, d, waste, capacity, n_original, curr_p, curr_c, pos
                )
                if improved:
                    return new_t, p_new, c_new, True

            # Step 6: Recurse to depth k+1
            res_tour, res_p, res_c, res_imp = backtrack(k + 1, t_list + [t_next, t_neighbor], new_gain_c)
            if res_imp:
                return res_tour, res_p, res_c, True

        return None, 0.0, 0.0, False

    # Start recursion from k=1: existing broken edge (t1, t2)
    return backtrack(1, [t1, t2], d[t1, t2])


def _verify_and_construct(
    curr_tour: List[int],
    t_list: List[int],
    d: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    n_original: Optional[int],
    curr_p: float,
    curr_c: float,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """Helper to verify k-opt closure and build resulting tour.

    Args:
        curr_tour: The current tour.
        t_list: Sequence of k-opt points.
        d: Distance matrix.
        waste: Waste.
        capacity: Capacity.
        n_original: Number of original nodes.
        curr_p: Current penalty.
        curr_c: Current cost.
        pos: Position of nodes.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improving move found.
    """
    k = len(t_list) // 2
    if k < 2:
        return None, 0.0, 0.0, False

    # Get positions of t_i in current tour
    try:
        if pos is None:
            return None, 0.0, 0.0, False
        _pos = pos
        pos_list = [int(_pos[node]) for node in t_list]
    except (KeyError, IndexError, TypeError):
        return None, 0.0, 0.0, False

    # Sequential k-opt added edges (in t_list indexing): (1,2), (3,4) ... (2k-1, 0)
    added_edges_indices = []
    for i in range(k):
        added_edges_indices.append((2 * i + 1, (2 * i + 2) % (2 * k)))

    try:
        new_tour = build_tour_from_segments(curr_tour, pos_list, added_edges_indices, k=k)
        p, c = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p, c, curr_p, curr_c):
            return new_tour, p, c, True
    except (ValueError, IndexError):
        # build_tour_from_segments may fail if t_list doesn't form a Hamiltonian cycle
        pass
    return None, 0.0, 0.0, False
