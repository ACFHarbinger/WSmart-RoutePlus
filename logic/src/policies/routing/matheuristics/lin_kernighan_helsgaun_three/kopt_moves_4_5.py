"""4-opt and 5-opt move routines with gain pre-screening (LKH-3).

Holds ``_try_4opt_move`` and ``_try_5opt_move``, split out of
``tour_improvement``.
"""

from __future__ import annotations

from random import Random
from typing import List, Optional, Tuple

import numpy as np

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.kopt_moves_2_3 import (
    check_acceptance,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.kopt_topologies import (
    EXHAUSTIVE_4OPT_CASES,
    EXHAUSTIVE_5OPT_CASES,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.load_tracker import (
    LoadState,
    get_exact_penalty_delta,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.objective import (
    get_score,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_construction import (
    build_tour_from_segments,
)


def _try_4opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    k: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    pos: Optional[np.ndarray] = None,
    acceptance_criterion: Optional[IAcceptanceCriterion] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 4-opt move with exhaustive topology enumeration.

    **ARCHITECTURE**: Uses module-level cached EXHAUSTIVE_4OPT_CASES constant
    (25 valid non-sequential topologies) with lazy evaluation pattern.

    Algorithm (Lazy Evaluation Pattern):
    1. For each fourth cut position l:
       a. Extract the 8 broken nodes [t1, t2, t3, t4, t5, t6, t7, t8]
       b. Iterate over ALL 25 cached 4-opt topologies
       c. Map cached indices [0..7] to actual nodes
       d. Compute added_cost and delta_c in O(1) using distance matrix
       e. Track best_delta_c and best_topology across all 25 cases
    2. LAZY PENALTY EVALUATION: Only after finding best topology
       a. Map topology indices to actual (broken_edges, added_edges)
       b. Call get_exact_penalty_delta for O(L) capacity check
    3. Apply lexicographic gate: ΔP < 0 OR (ΔP ≈ 0 AND ΔC > 0)
    4. ONLY if gate passes: construct tour array via build_tour_from_segments
    5. Verify with full (penalty, cost) evaluation

    Args:
        curr_tour: The current tour.
        i: The index of the first cut position.
        j: The index of the second cut position.
        k: The index of the third cut position.
        t1: The first node.
        t2: The second node.
        t3: The third node.
        t4: The fourth node.
        t5: The fifth node.
        t6: The sixth node.
        distance_matrix: Distance matrix.
        waste: Waste.
        capacity: Capacity.
        rng: Random number generator.
        n_original: Number of original nodes.
        load_state: Load state.
        pos: Position of nodes.
        acceptance_criterion: Acceptance criterion for moves.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improvement found.

    Complexity:
        - Per fourth-cut position: O(25) = O(1) topology evaluations
        - Distance lookups: O(25 × 4) = O(1) matrix accesses
        - Penalty delta: O(L) once per promising position
        - Tour construction: O(N) once per accepted move
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)
    for l in range(k + 2, nodes_count):
        t7 = curr_tour[l]
        t8 = curr_tour[(l + 1) % nodes_count]

        broken_nodes = [t1, t2, t3, t4, t5, t6, t7, t8]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8]

        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        for topology in EXHAUSTIVE_4OPT_CASES:
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]
            added_cost = sum(d[u, v] for u, v in added_edges)
            delta_c = base_cost - added_cost

            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        if best_topology is None:
            continue

        # CRITICAL FIX: REMOVED early exit for SA

        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,  # type: ignore[arg-type]
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Phase 3: Lexicographic gate using the MODULAR ACCEPTANCE CRITERION
        if not check_acceptance(delta_p, best_delta_c, curr_c, acceptance_criterion):
            continue

        # Phase 4: Construct tour array
        pos_indices = [i, i + 1, j, j + 1, k, k + 1, l, l + 1 if l + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos_indices, best_topology, k=4)

        # Phase 5: Full verification
        p4, c4 = get_score(new_tour, d, waste, capacity, n_original)

        return new_tour, p4, c4, True

    return None, 0.0, 0.0, False


def _try_5opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    k: int,
    l: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
    t5: int,
    t6: int,
    t7: int,
    t8: int,
    distance_matrix: np.ndarray,
    waste: Optional[np.ndarray],
    capacity: Optional[float],
    rng: Random,
    n_original: Optional[int] = None,
    load_state: Optional[LoadState] = None,
    pos: Optional[np.ndarray] = None,
    acceptance_criterion: Optional[IAcceptanceCriterion] = None,
) -> Tuple[Optional[List[int]], float, float, bool]:
    """
    Search for an improving 5-opt move with exhaustive topology enumeration.

    **CRITICAL UPGRADE**: This function now evaluates ALL 208 valid non-sequential
    5-opt reconnection topologies using the module-level cached constant
    EXHAUSTIVE_5OPT_CASES. The previous implementation evaluated only 5
    representative cases, which was mathematically incomplete for publication.

    Algorithm:
    1. For each fifth cut position m:
       a. Extract the 10 broken nodes [t1, t2, ..., t10]
       b. Iterate over ALL 208 cached 5-opt topologies
       c. Map cached indices [0..9] to actual nodes
       d. Compute added_cost and delta_c in O(1) using distance matrix
       e. Track best_delta_c and best_topology across all 208 cases
    2. LAZY PENALTY EVALUATION: Only after finding best topology
       a. Map topology indices to actual (broken_edges, added_edges)
       b. Call get_exact_penalty_delta for O(L) capacity check
    3. Apply lexicographic gate: ΔP < 0 OR (ΔP ≈ 0 AND ΔC > 0)
    4. ONLY if gate passes: construct tour array via build_tour_from_segments
    5. Verify with full (penalty, cost) evaluation

    Complexity:
        - Per fifth-cut position: O(208) = O(1) topology evaluations
        - Distance lookups: O(208 × 5) = O(1) matrix accesses
        - Penalty delta: O(L) once per promising position
        - Tour construction: O(N) once per accepted move
        - Overall: O(N^5) amortized (same asymptotic as before, more thorough)

    Args:
        curr_tour: The current tour.
        i: The index of the first cut position.
        j: The index of the second cut position.
        k: The index of the third cut position.
        l: The index of the fourth cut position.
        t1: The first node.
        t2: The second node.
        t3: The third node.
        t4: The fourth node.
        t5: The fifth node.
        t6: The sixth node.
        t7: The seventh node.
        t8: The eighth node.
        distance_matrix: Distance matrix.
        waste: Waste.
        capacity: Capacity.
        rng: Random number generator.
        n_original: Number of original nodes.
        load_state: Load state.
        pos: Position of nodes.
        acceptance_criterion: Acceptance criterion for moves.

    Returns:
        (new_tour, penalty, cost, improved) tuple.
        Returns (None, 0.0, 0.0, False) if no improving move found.

    References:
        Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
        traveling salesman heuristic. European Journal of Operational Research,
        126(1), 106-130.
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    for m in range(l + 2, nodes_count):
        t9 = curr_tour[m]
        t10 = curr_tour[(m + 1) % nodes_count]

        broken_nodes = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6), (t7, t8), (t9, t10)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6] + d[t7, t8] + d[t9, t10]

        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        for topology in EXHAUSTIVE_5OPT_CASES:
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]
            added_cost = sum(d[u, v] for u, v in added_edges)
            delta_c = base_cost - added_cost

            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        if best_topology is None:
            continue

        # CRITICAL FIX: REMOVED early exit for SA

        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                best_added_edges,  # type: ignore[arg-type]
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Phase 3: Lexicographic gate using the MODULAR ACCEPTANCE CRITERION
        if not check_acceptance(delta_p, best_delta_c, curr_c, acceptance_criterion):
            continue

        # Phase 4: Construct tour array
        pos_indices = [i, i + 1, j, j + 1, k, k + 1, l, l + 1, m, m + 1 if m + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, pos_indices, best_topology, k=5)

        # Phase 5: Full verification
        p5, c5 = get_score(new_tour, d, waste, capacity, n_original)

        return new_tour, p5, c5, True

    return None, 0.0, 0.0, False
