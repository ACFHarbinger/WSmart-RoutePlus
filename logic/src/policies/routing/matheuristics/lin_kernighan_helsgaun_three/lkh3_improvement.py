"""Main LKH-3 tour-improvement pass.

Holds ``_improve_tour``, the outer driver of one complete sequential k-opt
local-search pass (k = 2..5), split out of ``lkh3``.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.load_tracker import (
    build_load_state,
    update_load_state_after_move,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.objective import (
    is_better,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_improvement import (
    _dynamic_kopt_search,
    _try_2opt_move,
    _try_3opt_move,
    _try_4opt_move,
    _try_5opt_move,
    _try_oropt_move,
)


def _improve_tour(  # noqa: C901
    curr_tour: List[int],
    curr_pen: float,
    curr_cost: float,
    candidates: Dict[int, List[int]],
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    dont_look_bits: Optional[np.ndarray] = None,
    max_k_opt: int = 5,
    n_original: Optional[int] = None,
    dynamic_topology_discovery: bool = False,
    acceptance_criterion: Optional[IAcceptanceCriterion] = None,
) -> Tuple[List[int], float, float, bool, Optional[np.ndarray]]:
    """
    Execute one complete pass of sequential k-opt local search (k = 2..5).

    For each node t1 (not masked by a don't-look bit) and its successor t2
    in the current tour, the following hierarchy is attempted in order:

    1. **Dynamic Search** — if dynamic_topology_discovery is True, uses recursive B&B.
    2. **Exhaustive Hierarchy** — 2-opt, 3-opt, 4-opt, 5-opt patterns.
       move applied via _try_2opt_move with lazy lexicographic gate.
    2. **3-opt** — exact gain pre-screen for all seven patterns; move applied
       via :func:`move_kopt_intra` (k=3).  Restricted to n < 500.
    3. **4-opt** — exact gain pre-screen for three patterns; move applied via
       :func:`move_kopt_intra` (k=4).  Restricted to n < 300.
    4. **5-opt** — exact gain pre-screen for five patterns; move applied via
       :func:`move_kopt_intra` (k=5).  Restricted to n < 200.

    **Don't-look bit policy**: After an accepted k-opt move, only the nodes
    whose tour-adjacency physically changed have their don't-look bits reset.
    For 2-opt on (t1,t2,t3,t4): only t1 and t3 are new edge endpoints.

    Args:
        curr_tour: Current closed tour.
        curr_pen: Current penalty.
        curr_cost: Current cost.
        candidates: α-nearest-neighbour candidate lists.
        distance_matrix: Cost matrix.
        waste: Waste for each node.
        capacity: Capacity of the vehicle.
        rng: Random number generator forwarded to operators.
        dont_look_bits: Boolean array; nodes with True are skipped.
            Allocated up to max(nodes_count, n_original + n_vehicles) to
            avoid IndexError on augmented graphs.
        max_k_opt: Maximum k for k-opt moves (2-5).
        n_original: Original graph size before augmentation.
        dynamic_topology_discovery: Whether to discover k-opt topologies dynamically.
        acceptance_criterion: Acceptance criterion for moves.

    Returns:
        Tuple[List[int], float, float, bool, Optional[np.ndarray]]: New tour,
            updated penalty, updated cost, improvement flag, and updated bits.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # Allocate dont_look_bits safely for augmented graphs
    if dont_look_bits is None:
        max_idx = max(nodes_count, len(distance_matrix))
        dont_look_bits = np.zeros(max_idx, dtype=bool)
    elif len(dont_look_bits) < nodes_count:
        expanded = np.zeros(nodes_count, dtype=bool)
        expanded[: len(dont_look_bits)] = dont_look_bits
        dont_look_bits = expanded

    # Build load state for exact penalty queries (VRP mode only)
    load_state = None
    if waste is not None and capacity is not None and n_original is not None:
        load_state = build_load_state(curr_tour, waste, capacity, n_original)

    # Feature 1: Position lookup array (O(1) lookups instead of list.index)
    # Size to distance_matrix to handle augmented dummy depots safely.
    pos = np.full(len(distance_matrix), -1, dtype=np.int32)
    for idx, node in enumerate(curr_tour[:-1]):
        pos[node] = idx

    improved_overall = False
    for i in range(nodes_count):
        t1 = curr_tour[i]

        # Guard against IndexError on augmented dummy node IDs
        if t1 < len(dont_look_bits) and dont_look_bits[t1]:
            continue

        t2 = curr_tour[i + 1]
        found_improvement_for_t1 = False

        # ---- Dynamic Discovery (Recursive B&B) ----
        if dynamic_topology_discovery:
            res_dyn, rp_dyn, rc_dyn, improved_dyn = _dynamic_kopt_search(
                curr_tour,
                i,
                t1,
                t2,
                candidates,
                d,
                waste,
                capacity,
                rng,
                n_original=n_original,
                load_state=load_state,
                max_k=max_k_opt,
                pos=pos,
            )
            if improved_dyn and res_dyn is not None:
                curr_tour, curr_pen, curr_cost = res_dyn, rp_dyn, rc_dyn
                improved_overall = True
                found_improvement_for_t1 = True
                # Rebuild pos after accepted move
                for idx, node in enumerate(curr_tour[:-1]):
                    pos[node] = idx
                return curr_tour, curr_pen, curr_cost, True, dont_look_bits

            # If not improved in dynamic mode, set dont-look bit and skip exhaustive k-opt
            if t1 < len(dont_look_bits):
                dont_look_bits[t1] = True
            continue

        # ---- Or-opt (Segment Relocation 1, 2, 3 nodes) ----
        res_or, rp_or, rc_or, imp_or = _try_oropt_move(
            curr_tour,
            t1,
            i,
            candidates,
            d,
            waste,
            capacity,
            rng,
            n_original=n_original,
            load_state=load_state,
            pos=pos,
        )
        if imp_or and res_or is not None:
            curr_tour, curr_pen, curr_cost = res_or, rp_or, rc_or
            improved_overall = True
            found_improvement_for_t1 = True
            if t1 < len(dont_look_bits):
                dont_look_bits[t1] = False
            # Rebuild pos after accepted move
            for idx, node in enumerate(curr_tour[:-1]):
                pos[node] = idx
            return curr_tour, curr_pen, curr_cost, True, dont_look_bits

        # ---- Exhaustive 2-opt ----
        res_tour, res_p, res_c, res_imp, j = _try_2opt_move(
            curr_tour,
            i,
            t1,
            t2,
            candidates,
            d,
            waste,
            capacity,
            rng,
            n_original=n_original,
            load_state=load_state,
            pos=pos,
            acceptance_criterion=acceptance_criterion,
        )
        if res_imp and res_tour is not None:
            curr_tour, curr_pen, curr_cost = res_tour, res_p, res_c
            improved_overall = True
            found_improvement_for_t1 = True
            t3 = curr_tour[j] if j >= 0 else t1
            if t1 < len(dont_look_bits):
                dont_look_bits[t1] = False
            if t3 < len(dont_look_bits):
                dont_look_bits[t3] = False

            if load_state is not None:
                load_state = update_load_state_after_move(
                    load_state,
                    curr_tour,
                    waste,  # type: ignore[arg-type]
                    capacity,  # type: ignore[arg-type]
                    n_original,  # type: ignore[arg-type]
                )
            for idx, node in enumerate(curr_tour[:-1]):
                pos[node] = idx
            return curr_tour, curr_pen, curr_cost, True, dont_look_bits

        # ---- 3-opt ----
        if nodes_count < 500 and max_k_opt >= 3:
            for t3_cand in candidates[t2]:
                if t3_cand == t1 or (t2 < len(distance_matrix) and pos[t3_cand] < 0):
                    continue
                j = int(pos[t3_cand])
                if j <= i + 1:
                    continue
                t4 = curr_tour[j + 1]

                res_tour, res_p, res_c, res_imp = _try_3opt_move(
                    curr_tour,
                    i,
                    j,
                    t1,
                    t2,
                    t3_cand,
                    t4,
                    d,
                    waste,
                    capacity,
                    rng,
                    n_original=n_original,
                    load_state=load_state,
                    pos=pos,
                    acceptance_criterion=acceptance_criterion,
                )
                if res_imp and res_tour is not None and is_better(res_p, res_c, curr_pen, curr_cost):
                    curr_tour, curr_pen, curr_cost = res_tour, res_p, res_c
                    improved_overall = True
                    found_improvement_for_t1 = True
                    # Reset only new edge endpoints
                    for nd in [t1, t3_cand]:
                        if nd < len(dont_look_bits):
                            dont_look_bits[nd] = False
                    if load_state is not None:
                        load_state = update_load_state_after_move(
                            load_state,
                            curr_tour,
                            waste,  # type: ignore[arg-type]
                            capacity,  # type: ignore[arg-type]
                            n_original,  # type: ignore[arg-type]
                        )
                    # Rebuild pos after accepted move
                    for idx, node in enumerate(curr_tour[:-1]):
                        pos[node] = idx
                    return curr_tour, curr_pen, curr_cost, True, dont_look_bits

                # ---- 4-opt ----
                if nodes_count < 300 and max_k_opt >= 4:
                    for k_idx in range(j + 2, nodes_count):
                        t5 = curr_tour[k_idx]
                        t6 = curr_tour[(k_idx + 1) % nodes_count]

                        res4, rp4, rc4, ri4 = _try_4opt_move(
                            curr_tour,
                            i,
                            j,
                            k_idx,
                            t1,
                            t2,
                            t3_cand,
                            t4,
                            t5,
                            t6,
                            d,
                            waste,
                            capacity,
                            rng,
                            n_original=n_original,
                            load_state=load_state,
                            pos=pos,
                            acceptance_criterion=acceptance_criterion,
                        )
                        if ri4 and res4 is not None and is_better(rp4, rc4, curr_pen, curr_cost):
                            curr_tour, curr_pen, curr_cost = res4, rp4, rc4
                            improved_overall = True
                            found_improvement_for_t1 = True
                            for nd in [t1, t3_cand, t5]:
                                if nd < len(dont_look_bits):
                                    dont_look_bits[nd] = False
                            if load_state is not None:
                                load_state = update_load_state_after_move(
                                    load_state,
                                    curr_tour,
                                    waste,  # type: ignore[arg-type]
                                    capacity,  # type: ignore[arg-type]
                                    n_original,  # type: ignore[arg-type]
                                )
                            # Rebuild pos after accepted move
                            for idx, node in enumerate(curr_tour[:-1]):
                                pos[node] = idx
                            return curr_tour, curr_pen, curr_cost, True, dont_look_bits

                        # ---- 5-opt ----
                        if nodes_count < 200 and max_k_opt >= 5:
                            for l_idx in range(k_idx + 2, nodes_count):
                                t7 = curr_tour[l_idx]
                                t8 = curr_tour[(l_idx + 1) % nodes_count]
                                res5, rp5, rc5, ri5 = _try_5opt_move(
                                    curr_tour,
                                    i,
                                    j,
                                    k_idx,
                                    l_idx,
                                    t1,
                                    t2,
                                    t3_cand,
                                    t4,
                                    t5,
                                    t6,
                                    t7,
                                    t8,
                                    d,
                                    waste,
                                    capacity,
                                    rng,
                                    n_original=n_original,
                                    load_state=load_state,
                                    pos=pos,
                                    acceptance_criterion=acceptance_criterion,
                                )
                                if ri5 and res5 is not None and is_better(rp5, rc5, curr_pen, curr_cost):
                                    curr_tour, curr_pen, curr_cost = res5, rp5, rc5
                                    improved_overall = True
                                    found_improvement_for_t1 = True
                                    for nd in [t1, t3_cand, t5, t7]:
                                        if nd < len(dont_look_bits):
                                            dont_look_bits[nd] = False
                                    if load_state is not None:
                                        load_state = update_load_state_after_move(
                                            load_state,
                                            curr_tour,
                                            waste,  # type: ignore[arg-type]
                                            capacity,  # type: ignore[arg-type]
                                            n_original,  # type: ignore[arg-type]
                                        )
                                    # Rebuild pos after accepted move
                                    for idx, node in enumerate(curr_tour[:-1]):
                                        pos[node] = idx
                                    return curr_tour, curr_pen, curr_cost, True, dont_look_bits

        if not found_improvement_for_t1 and t1 < len(dont_look_bits):
            dont_look_bits[t1] = True

    return curr_tour, curr_pen, curr_cost, improved_overall, dont_look_bits
