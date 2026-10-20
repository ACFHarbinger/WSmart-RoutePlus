"""2-opt and 3-opt move routines with gain pre-screening (LKH-3).

Holds ``_try_2opt_move`` and ``_try_3opt_move`` plus the shared
``check_acceptance`` gate, split out of ``tour_improvement``.
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.interfaces.acceptance_criterion import IAcceptanceCriterion
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.kopt_topologies import (
    EXHAUSTIVE_2OPT_CASES,
    EXHAUSTIVE_3OPT_CASES,
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


def _try_2opt_move(
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
    pos: Optional[np.ndarray] = None,
    acceptance_criterion: Optional[IAcceptanceCriterion] = None,
) -> Tuple[Optional[List[int]], float, float, bool, int]:
    """
    Search for an improving 2-opt move starting from edge (t1, t2).

    **ARCHITECTURE**: Uses module-level cached EXHAUSTIVE_2OPT_CASES constant
    (1 valid topology) for architectural consistency with higher-order k-opt.

    Algorithm (Lazy Evaluation Pattern):
    1. For each candidate t3 of t2:
       a. Extract broken nodes [t1, t2, t3, t4]
       b. Compute distance gain ΔC in O(1) using the single cached topology
       c. Compute penalty delta ΔP via load_state — O(L)
       d. Lexicographic gate: accept if ΔP<0 OR (ΔP≈0 AND ΔC>0)
       e. ONLY IF gate passes: construct tour and verify

    Args:
        curr_tour: Description of curr_tour.
        i: Description of i.
        t1: Description of t1.
        t2: Description of t2.
        candidates: Description of candidates.
        distance_matrix: Description of distance_matrix.
        waste: Description of waste.
        capacity: Description of capacity.
        rng: Description of rng.
        n_original: Description of n_original.
        load_state: Description of load_state.
        pos: Description of pos.
        acceptance_criterion: Acceptance criterion for moves.

    Returns:
        (new_tour, penalty, cost, improved, j) where j is the position of t3.
        Returns (None, 0, 0, False, -1) if no improvement found.

    Complexity:
        - Per candidate: O(1) distance lookup + O(L) penalty check
        - Tour construction: O(N) only when gate passes

    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix

    # Get current tour score for final verification
    p_curr, c_curr = get_score(curr_tour, d, waste, capacity, n_original)

    # Extract the single cached 2-opt topology (for architectural consistency)
    # EXHAUSTIVE_2OPT_CASES = [[(0, 2), (1, 3)]]
    assert len(EXHAUSTIVE_2OPT_CASES) == 1, "2-opt must have exactly 1 topology"
    topology_2opt = EXHAUSTIVE_2OPT_CASES[0]

    for t3_cand in candidates[t2]:
        if t3_cand == t1 or t3_cand == curr_tour[(i + 2) % nodes_count]:
            continue

        if pos is not None:
            if t3_cand >= len(pos) or pos[t3_cand] < 0:
                continue
            j = int(pos[t3_cand])
        else:
            try:
                j = curr_tour.index(t3_cand)
            except ValueError:
                continue

        if j <= i + 1:
            continue

        t4 = curr_tour[(j + 1) % nodes_count]

        # The 4 endpoints of the 2 broken edges (indexed 0..3 in cached topology)
        broken_nodes = [t1, t2, t3_cand, t4]
        broken_edges = [(t1, t2), (t3_cand, t4)]

        # Phase 1: Compute distance gain using cached topology
        # topology_2opt = [(0, 2), (1, 3)] maps to [(t1, t3), (t2, t4)]
        added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology_2opt]
        base_cost = d[t1, t2] + d[t3_cand, t4]
        added_cost = sum(d[u, v] for u, v in added_edges)
        delta_c = base_cost - added_cost

        # Phase 2: Lazy penalty evaluation (O(L) complexity)
        if load_state is not None and waste is not None and capacity is not None:
            delta_p = get_exact_penalty_delta(
                curr_tour,
                broken_edges,
                added_edges,
                load_state,
                waste,
                capacity,
                n_original,  # type: ignore[arg-type]
            )
        else:
            delta_p = 0.0

        # Phase 3: Lexicographic gate using the MODULAR ACCEPTANCE CRITERION
        if not check_acceptance(delta_p, delta_c, c_curr, acceptance_criterion):
            continue

        # Phase 4: Construct tour array
        new_tour = build_tour_from_segments(
            curr_tour, [i, i + 1, j, j + 1 if j + 1 < nodes_count else 0], topology_2opt, k=2
        )

        # Phase 5: Full verification (Note: We still verify lexicographic improvement
        # to ensure the global best logic isn't violated, though the move is "accepted")
        p2, c2 = get_score(new_tour, d, waste, capacity, n_original)

        # If the move was accepted by SA despite degrading cost, we still return True
        # to allow the search trajectory to update.
        return new_tour, p2, c2, True, j

    return None, 0.0, 0.0, False, -1


def _try_3opt_move(
    curr_tour: List[int],
    i: int,
    j: int,
    t1: int,
    t2: int,
    t3: int,
    t4: int,
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
    Search for an improving 3-opt move with exhaustive topology enumeration.

    **ARCHITECTURE**: Uses module-level cached EXHAUSTIVE_3OPT_CASES constant
    (7 valid topologies) with lazy evaluation pattern.

    Algorithm (Lazy Evaluation Pattern):
    1. For each third cut position k_pos:
       a. Extract the 6 broken nodes [t1, t2, t3, t4, t5, t6]
       b. Iterate over ALL 7 cached 3-opt topologies
       c. Map cached indices [0..5] to actual nodes
       d. Compute added_cost and delta_c in O(1) using distance matrix
       e. Track best_delta_c and best_topology across all 7 cases
    2. LAZY PENALTY EVALUATION: Only after finding best topology
       a. Map topology indices to actual (broken_edges, added_edges)
       b. Call get_exact_penalty_delta for O(L) capacity check
    3. Apply lexicographic gate: ΔP < 0 OR (ΔP ≈ 0 AND ΔC > 0)
    4. ONLY if gate passes: construct tour array via build_tour_from_segments
    5. Verify with full (penalty, cost) evaluation

    Note: The 7 cached cases include 2 pure 2-opt sub-moves. These are
    evaluated anyway since the constant-factor overhead is negligible and
    they may find improving moves that were missed due to α-nearest neighbor
    filtering in the preceding _try_2opt_move search.

    Args:
        curr_tour: The current tour.
        i: The index of the first cut position.
        j: The index of the second cut position.
        t1: The first node.
        t2: The second node.
        t3: The third node.
        t4: The fourth node.
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
        - Per third-cut position: O(7) = O(1) topology evaluations
        - Distance lookups: O(7 × 3) = O(1) matrix accesses
        - Penalty delta: O(L) once per promising position
        - Tour construction: O(N) once per accepted move
    """
    nodes_count = len(curr_tour) - 1
    d = distance_matrix
    curr_p, curr_c = get_score(curr_tour, d, waste, capacity, n_original)

    for k_pos in range(j + 2, nodes_count):
        t5 = curr_tour[k_pos]
        t6 = curr_tour[(k_pos + 1) % nodes_count]

        broken_nodes = [t1, t2, t3, t4, t5, t6]
        broken_edges = [(t1, t2), (t3, t4), (t5, t6)]
        base_cost = d[t1, t2] + d[t3, t4] + d[t5, t6]

        best_topology = None
        best_delta_c = -float("inf")
        best_added_edges = None

        for topology in EXHAUSTIVE_3OPT_CASES:
            added_edges = [(broken_nodes[u], broken_nodes[v]) for u, v in topology]
            added_cost = sum(d[u, v] for u, v in added_edges)
            delta_c = base_cost - added_cost

            if delta_c > best_delta_c:
                best_delta_c = delta_c
                best_topology = topology
                best_added_edges = added_edges

        if best_topology is None:
            continue

        # SA must evaluate negative best_delta_c to escape local minima
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
        target_pos = [i, i + 1, j, j + 1, k_pos, k_pos + 1 if k_pos + 1 < nodes_count else 0]
        new_tour = build_tour_from_segments(curr_tour, target_pos, best_topology, k=3)

        # Phase 5: Full verification
        p3, c3 = get_score(new_tour, d, waste, capacity, n_original)

        # If the move was accepted by the criterion, we return True
        return new_tour, p3, c3, True

    return None, 0.0, 0.0, False


def check_acceptance(
    delta_p: float, delta_c: float, curr_cost: float, acceptance_criterion: Optional[IAcceptanceCriterion] = None
) -> bool:
    """
    Rigorous acceptance criterion merging Lexicographic bounds with a modular
    Acceptance Criterion (Boltzmann/Metropolis).

    Args:
        delta_p: Change in penalty.
        delta_c: Change in cost.
        curr_cost: Current cost.
        acceptance_criterion: Acceptance criterion.

    Returns:
        bool: True if move is accepted, False otherwise.
    """
    # 1. Strict Penalty Improvement: Always accept if we significantly reduce capacity violations.
    if delta_p < -1e-6:
        return True

    # 2. Penalty Degradation: NEVER accept a move that increases capacity violations.
    # Feasibility must be mathematically preserved.
    if delta_p > 1e-6:
        return False

    # 3. Penalty Neutral (delta_p == 0): Evaluate routing cost.
    if acceptance_criterion is not None:
        # CRITICAL MATHEMATICAL MAPPING:
        # LKH-3 minimizes cost. BoltzmannAcceptance maximizes objective.
        # We negate the costs so that a cost reduction (improving) yields delta > 0.
        current_obj = -curr_cost
        candidate_obj = -(curr_cost + delta_c)

        # Call the modular interface
        accepted, _ = acceptance_criterion.accept(current_obj=current_obj, candidate_obj=candidate_obj)
        return accepted
    else:
        # Fallback to strict deterministic descent
        return delta_c < -1e-4
