"""Or-opt (segment relocation) move routines for LKH-3.

Holds ``__or_opt_relocation`` and ``_try_oropt_move``, split out of
``tour_improvement``.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.load_tracker import (
    LoadState,
    get_exact_penalty_delta,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.objective import (
    get_score,
    is_better,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_construction import (
    _should_accept_kopt_move,
)


def __or_opt_relocation(
    seg_len,
    t1,
    t_first,
    t_after,
    t_last,
    t_dest,
    t_dest_after,
    d,
    load_state,
    waste,
    capacity,
    curr_tour,
    n_original,
    i,
    nodes_count,
    curr_p,
    curr_c,
):
    """
    Evaluates and applies an Or-opt relocation move.

    Args:
        seg_len: Length of the segment to relocate.
        t1: Base node.
        t_first: First node of the segment.
        t_after: Node originally after the segment.
        t_last: Last node of the segment.
        t_dest: Destination node.
        t_dest_after: Node after the destination.
        d: Distance matrix.
        load_state: Load state.
        waste: Node wastes.
        capacity: Vehicle capacity.
        curr_tour: Current tour.
        n_original: Original number of nodes.
        i: Current index in the tour.
        nodes_count: Number of nodes.
        curr_p: Current penalty.
        curr_c: Current cost.

    Returns:
        A tuple of (new_tour, penalty, cost, improved).
    """
    orientations = [True, False] if seg_len > 1 else [True]
    for forward in orientations:
        # Broken: (t1, t_first), (t_last, t_after), (t_dest, t_dest_after)
        # Added (Forward): (t1, t_after), (t_dest, t_first), (t_last, t_dest_after)
        # Added (Reverse): (t1, t_after), (t_dest, t_last), (t_first, t_dest_after)

        broken_edges = [(t1, t_first), (t_last, t_after), (t_dest, t_dest_after)]
        if forward:
            added_edges = [(t1, t_after), (t_dest, t_first), (t_last, t_dest_after)]
        else:
            added_edges = [(t1, t_after), (t_dest, t_last), (t_first, t_dest_after)]

        delta_c = (d[t1, t_first] + d[t_last, t_after] + d[t_dest, t_dest_after]) - (
            d[t1, t_after] + d[t_dest, added_edges[1][1]] + d[added_edges[2][0], t_dest_after]
        )

        if delta_c <= 1e-6:
            continue

        delta_p = 0.0
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour, broken_edges, added_edges, load_state, waste, capacity, n_original
            )

        if not _should_accept_kopt_move(delta_p, delta_c):
            continue

        # Construct new tour
        # Segment nodes: [(i+1)%nodes_count ... (i+seg_len)%nodes_count]
        seg = []
        for idx_s in range(i + 1, i + seg_len + 1):
            seg.append(curr_tour[idx_s % nodes_count])

        if not forward:
            seg.reverse()

        # Build the tour by removing the segment and re-inserting it
        # 1. Remove segment from original tour
        tour_no_seg = []
        # Segment indices to skip
        skip_indices = set((idx_s % nodes_count) for idx_s in range(i + 1, i + seg_len + 1))
        for idx_n in range(nodes_count):
            if idx_n not in skip_indices:
                tour_no_seg.append(curr_tour[idx_n])

        # 2. Re-insert after t_dest (which is now in tour_no_seg)
        try:
            insert_pos = tour_no_seg.index(t_dest)
            new_tour_open = tour_no_seg[: insert_pos + 1] + seg + tour_no_seg[insert_pos + 1 :]
            new_tour = new_tour_open + [new_tour_open[0]]
        except ValueError:
            continue  # Should not happen

        p_new, c_new = get_score(new_tour, d, waste, capacity, n_original)
        if is_better(p_new, c_new, curr_p, curr_c):
            return new_tour, p_new, c_new, True


def _try_oropt_move(
    curr_tour: List[int],
    t1: int,
    i: int,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    pos: Optional[np.ndarray] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Or-opt move: Relocate a segment of 1, 2, or 3 nodes.
    Relocates the segment [tour[i+1]...tour[i+len]] to follow node t_dest.
    Args:
        curr_tour: The current tour.
        t1: The first node.
        i: The index of the first cut position.
        candidates: Candidates.
        distance_matrix: Distance matrix.
        waste: Waste.
        capacity: Capacity.
        rng: Random number generator.
        n_original: Number of original nodes.
        load_state: Load state.
        pos: Position of nodes.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improving move found.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    # Segment lengths for Or-opt (standard LKH-3 uses 1, 2, 3)
    for seg_len in [3, 2, 1]:
        if nodes_count < seg_len + 3:
            continue

        # Segment nodes: [t_first, ..., t_last]
        # Current edges to break: (t1, t_first) and (t_last, t_after)
        t_first = curr_tour[(i + 1) % nodes_count]
        t_last = curr_tour[(i + seg_len) % nodes_count]
        t_after = curr_tour[(i + seg_len + 1) % nodes_count]

        # Try inserting this segment after each possible destination t_dest
        # We use candidates of t_first and t_last to find promising t_dest
        potential_dest_nodes = set()
        for cand in candidates[t_first]:
            potential_dest_nodes.add(cand)
        for cand in candidates[t_last]:
            potential_dest_nodes.add(cand)

        for t_dest in potential_dest_nodes:
            if t_dest in curr_tour[i + 1 : i + seg_len + 1]:
                continue

            # Find position of t_dest
            if pos is not None:
                if t_dest >= len(pos) or pos[t_dest] < 0:
                    continue
                dest_idx = int(pos[t_dest])
            else:
                try:
                    dest_idx = curr_tour.index(t_dest)
                except ValueError:
                    continue

            t_dest_after = curr_tour[(dest_idx + 1) % nodes_count]
            if t_dest_after == t_first:
                continue

            # Or-opt relocation can happen in two orientations (if seg_len > 1)
            # 1. Forward: ... t_dest, t_first ... t_last, t_dest_after ...
            # 2. Reverse: ... t_dest, t_last ... t_first, t_dest_after ...
            __or_opt_relocation(
                seg_len,
                t1,
                t_first,
                t_after,
                t_last,
                t_dest,
                t_dest_after,
                d,
                load_state,
                waste,
                capacity,
                curr_tour,
                n_original,
                i,
                nodes_count,
                curr_p,
                curr_c,
            )
    return None, 0.0, 0.0, False
