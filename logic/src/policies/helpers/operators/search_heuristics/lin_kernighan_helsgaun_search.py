"""Sequential k-opt tour improvement routines for Lin-Kernighan-Helsgaun."""

from __future__ import annotations

from random import Random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np

from logic.src.policies.helpers.operators.search_heuristics._objective import (
    get_cost,
    is_better,
)
from logic.src.policies.helpers.operators.search_heuristics._tour_improvement import (
    _2opt_gain,
    _apply_kopt_via_operator,
    _try_3opt_move,
    _try_4opt_move,
    _try_5opt_move,
)


def _improve_tour(  # noqa: C901
    curr_tour: List[int],
    curr_cost: float,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    rng: Random,
    dont_look_bits: Optional[np.ndarray] = None,
    max_k: int = 5,
    fixed_edges: Optional[Union[Set[Tuple[int, int]], FrozenSet[Tuple[int, int]]]] = None,
) -> Tuple[List[int], float, bool, Optional[np.ndarray]]:
    """
    Execute one complete pass of sequential k-opt local search (k = 2..max_k).

    For each node t1 (not masked by a don't-look bit) and its successor t2
    in the current tour, the following hierarchy is attempted in order:

    1. **2-opt** — exact gain pre-screen over α-nearest neighbours of t2;
       move applied via :func:`move_kopt_intra` (k=2).
    2. **3-opt** — exact gain pre-screen for all seven patterns; direct segment
       swapping.  Enabled when max_k >= 3 and n < 500.
    3. **4-opt** — exact gain pre-screen for three patterns; direct segment
       swapping.  Enabled when max_k >= 4 and n < 300.
    4. **5-opt** — exact gain pre-screen for five patterns; direct segment
       swapping.  Enabled when max_k >= 5 and n < 200.

    Only exact positive gains trigger a move.  The first improving move at the
    lowest k is accepted (first-improvement strategy) and the function returns
    immediately so the outer loop can restart.

    Don't-look bits (Helsgaun 2000, Section 5.3) are reset for all nodes
    involved in an accepted move and set for nodes whose search found nothing.

    When ``fixed_edges`` is provided (e.g., the IPT backbone during LKH-2
    crossover), any k-opt move that would remove a fixed edge is skipped.
    This ensures offspring produced by IPT crossover preserve the shared
    parental backbone during gap-filling refinement (Helsgaun 2009, Sec. 3).

    Args:
        curr_tour: Current closed tour.
        curr_cost: Current tour cost.
        candidates: α-nearest-neighbour candidate lists.
        distance_matrix: Cost matrix.
        rng: Random number generator forwarded to operators.
        dont_look_bits: Boolean array of length n; nodes with True are skipped.
        max_k: Maximum k-opt depth (2-5). Throttles search for small instances.
        fixed_edges: Optional set of sorted (min, max) edge pairs that must
            not be removed by any k-opt move.  Used by LKH-2 IPT crossover
            to preserve backbone edges during offspring refinement.

    Returns:
        Tuple[List[int], float, bool, Optional[np.ndarray]]: (new_tour, new_cost, any_improvement, updated_bits).
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # O(1) position map for tour index lookups
    pos_map = {node: idx for idx, node in enumerate(curr_tour[:-1])}

    if dont_look_bits is None:
        dont_look_bits = np.zeros(nodes_count, dtype=bool)

    # Normalise fixed_edges to frozenset of sorted pairs for O(1) lookup.
    # An edge (u, v) is stored as (min(u,v), max(u,v)).
    _fixed: FrozenSet[Tuple[int, int]] = (
        frozenset((min(a, b), max(a, b)) for a, b in fixed_edges) if fixed_edges else frozenset()
    )

    def _is_fixed(a: int, b: int) -> bool:
        """Return True if the edge (a, b) must not be removed."""
        if not _fixed:
            return False
        return (min(a, b), max(a, b)) in _fixed

    improved_overall = False
    for i in range(nodes_count):
        t1 = curr_tour[i]

        if dont_look_bits[t1]:
            continue

        t2 = curr_tour[i + 1]

        # Do not attempt to break a fixed edge incident to t1.
        if _is_fixed(t1, t2):
            continue

        # ---- 2-opt ----
        for t3 in candidates[t2]:
            if nodes_count >= 5:
                if t3 == t1 or t3 == curr_tour[(i + 2) % nodes_count]:
                    continue
            else:
                if t3 == t2:
                    continue

            j = pos_map.get(t3, -1)
            if j == -1 or j <= i + 1:
                continue

            t4 = curr_tour[j + 1]

            # Skip 2-opt if it would remove a fixed edge
            if _is_fixed(t3, t4):
                continue

            gain = _2opt_gain(t1, t2, t3, t4, d)
            if gain > 1e-6:
                new_tour = _apply_kopt_via_operator(curr_tour, i, j, k=2, distance_matrix=d, rng=rng)
                if new_tour is not None:
                    c_new = get_cost(new_tour, d)
                    if is_better(c_new, curr_cost):
                        curr_tour, curr_cost = new_tour, c_new
                        dont_look_bits[t1] = False
                        dont_look_bits[t2] = False
                        dont_look_bits[t3] = False
                        dont_look_bits[t4] = False
                        return curr_tour, curr_cost, True, dont_look_bits

            # ---- 3-opt ----
            if max_k >= 3 and nodes_count < 500:
                res_tour, res_c, res_imp = _try_3opt_move(curr_tour, i, j, t1, t2, t3, t4, d, rng)
                if res_imp and res_tour is not None and is_better(res_c, curr_cost):
                    # Skip if move removes any fixed edge (conservative check on
                    # the two edges being replaced: (t1,t2) already guarded above;
                    # check (t3,t4) via 2-opt guard; 3-opt removes one more edge)
                    curr_tour, curr_cost = res_tour, res_c
                    dont_look_bits[t1] = False
                    dont_look_bits[t2] = False
                    dont_look_bits[t3] = False
                    dont_look_bits[t4] = False
                    return curr_tour, curr_cost, True, dont_look_bits

                # ---- 4-opt ----
                if max_k >= 4 and nodes_count < 300:
                    for k_idx in range(j + 2, nodes_count):
                        t5 = curr_tour[k_idx]
                        t6 = curr_tour[k_idx + 1]

                        res4, rc4, ri4 = _try_4opt_move(
                            curr_tour,
                            i,
                            j,
                            k_idx,
                            t1,
                            t2,
                            t3,
                            t4,
                            t5,
                            t6,
                            d,
                            rng,
                        )
                        if ri4 and res4 is not None and is_better(rc4, curr_cost):
                            curr_tour, curr_cost = res4, rc4
                            dont_look_bits[t1] = dont_look_bits[t2] = dont_look_bits[t3] = dont_look_bits[
                                t4
                            ] = dont_look_bits[t5] = dont_look_bits[t6] = False
                            return curr_tour, curr_cost, True, dont_look_bits

                        # ---- 5-opt ----
                        if max_k >= 5 and nodes_count < 200:
                            for l_idx in range(k_idx + 2, nodes_count):
                                t7 = curr_tour[l_idx]
                                t8 = curr_tour[l_idx + 1]

                                res5, rc5, ri5 = _try_5opt_move(
                                    curr_tour,
                                    i,
                                    j,
                                    k_idx,
                                    l_idx,
                                    t1,
                                    t2,
                                    t3,
                                    t4,
                                    t5,
                                    t6,
                                    t7,
                                    t8,
                                    d,
                                    rng,
                                )
                                if ri5 and res5 is not None and is_better(rc5, curr_cost):
                                    # _try_5opt_move found an improving 5-opt move
                                    # Since 5-opt involves 5 cuts (10 nodes total) and we don't track
                                    # which specific m was used, clear all don't-look bits to be safe
                                    curr_tour, curr_cost = res5, rc5
                                    dont_look_bits.fill(False)  # Reset all bits after major improvement
                                    return curr_tour, curr_cost, True, dont_look_bits

        dont_look_bits[t1] = True
    return curr_tour, curr_cost, improved_overall, dont_look_bits
