"""
LKH-3 Heuristic Module.

Full implementation of the Lin-Kernighan-Helsgaun heuristic (LKH-3) for solving
the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP).

Key features (following Helsgaun 2000 and LKH-3 extension paper):

1. **Alpha-measure candidate sets** (Section 4.1, Helsgaun 2000):
   Edges pruned using sensitivity analysis on minimum spanning 1-trees. Only
   α-nearest neighbours are considered for inclusion in the tour.

2. **Subgradient optimisation / penalty vector** (Section 4.1):
   Node penalties π that transform C → D = cᵢⱼ + πᵢ + πⱼ, maximising the
   lower bound given by the minimum 1-tree length.

3. **Sequential k-opt moves (k = 2..5)** (Section 4.3):
   Exact gain computation for every reconnection case at each level, with
   early-termination via the positive-gain criterion.  Routines for higher-
   order moves (3 to 5-opt) are delegated to components in
   ``._tour_improvement``, with all final segment swaps executed via
   :func:`move_kopt_intra` from
   ``logic.src.policies.helpers.operators.intra_route.k_opt``.

4. **Candidate-set restricted search** (Section 3.2 / 4.1):
   Inner loops restricted to the α-nearest neighbours of each node, giving
   O(n · k_cand) search per node rather than O(n²).

5. **Don't-look bits** (Section 5.3):
   Nodes whose neighbourhood yielded no improvement are skipped until a
   neighbouring move changes their tour-adjacency.

6. **Double-bridge kick** (ILS perturbation):
   4-opt non-sequential perturbation that escapes local optima, delegated to
   :func:`double_bridge` from
   ``logic.src.policies.helpers.operators.perturbation.double_bridge``.

7. **Tour merging** (pool-based recombination):
   Combines shared edges from several elite tours to seed new searches.

8. **VRP penalty / lexicographic objective** (LKH-3):
   Objective = (penalty, cost); penalty = total excess demand. All
   improvement checks use ``is_better(p1, c1, p2, c2)``.

9. **_improve_tour** (in :mod:`lkh3_improvement`) — outer driver.  Iterates
   over every non-masked node t1 and tries 2-opt first; if unsuccessful it
   attempts 3-, 4-, and 5-opt in order (each gated by an instance-size
   threshold).  Accepts the first improvement found (first-improvement
   strategy) and returns immediately.  Uses *don't-look bits* (Helsgaun 2000,
   Section 5.3) to skip nodes whose neighbourhood was exhausted since the
   last move touched them.

References:
    Helsgaun, K. (2000). An effective implementation of the Lin-Kernighan
      traveling salesman heuristic. EJOR 126, 106-130.
    Helsgaun, K. (2017). An extension of the LKH-TSP solver for constrained
      traveling salesman and vehicle routing problems.

Attributes:
    solve_lkh3: Main LKH-3 solver entry point for TSP/CVRP instances.
    solve_lkh3_with_alns: LKH-3 solver with integrated ALNS post-processing.

Example:
    >>> import numpy as np
    >>> from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.lkh3 import solve_lkh3
    >>> dm = np.zeros((3, 3))
    >>> routes, cost, penalty = solve_lkh3(dm)
"""

from __future__ import annotations

import warnings
from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.graph_augmentation import (
    augment_graph,
    decode_augmented_tour,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.lkh3_improvement import (
    _improve_tour,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.objective import (
    compute_alpha_measures,
    get_candidate_set,
    get_score,
    is_better,
    solve_subgradient,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.popmusic import (
    popmusic_candidates,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_initialization import (
    _double_bridge_kick,
    _initialize_tour,
)
from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.tour_merging import merge_tours_best
from logic.src.tracking.viz_mixin import PolicyStateRecorder


def solve_lkh3(  # noqa: C901
    distance_matrix: np.ndarray,
    initial_tour: Optional[List[int]] = None,
    waste: Optional[np.ndarray] = None,
    capacity: float = 100.0,
    revenue: float = 1.0,
    cost_unit: float = 1.0,
    mandatory_nodes: Optional[List[int]] = None,
    coords: Optional[np.ndarray] = None,
    # LKH-3 parameters
    max_trials: int = 100,
    popmusic_subpath_size: int = 50,
    popmusic_trials: int = 50,
    popmusic_max_candidates: int = 5,
    max_k_opt: int = 5,
    use_ip_merging: bool = True,
    max_pool_size: int = 5,
    subgradient_iterations: int = 50,
    # LNS parameters
    profit_aware_operators: bool = False,
    alns_iterations: int = 10,
    plateau_limit: int = 10,
    deep_plateau_limit: int = 30,
    perturb_operator_weights: Optional[List[float]] = None,
    # Other parameters
    n_vehicles: int = 0,
    n_original: int = 0,
    candidate_set: Optional[Dict[int, List[int]]] = None,
    # System parameters
    recorder: Optional[PolicyStateRecorder] = None,
    np_rng: Optional[np.random.Generator] = None,
    rng: Optional[Random] = None,
    seed: int = 42,
    dynamic_topology_discovery: bool = False,
    native_prize_collecting: bool = False,
    sa_max_trials: int = 0,
) -> Tuple[List[List[int]], float, float]:
    """
    Solve a TSP or CVRP instance using the LKH-3 iterated local-search scheme.

    Algorithm outline (Helsgaun 2000 / LKH-3):

    1. **Initialisation** — nearest-neighbour or provided tour.
    2. **Candidate sets** — α-measure computed from minimum spanning tree.
    3. **Local-search loop** — repeated k-opt passes (k = 2..5) with
       don't-look bits until a local optimum is reached.  All k-opt moves
       are executed via :func:`move_kopt_intra`.
    4. **Perturbation** — double-bridge kick via :func:`double_bridge` to
       escape the basin of attraction of the current local optimum.
    5. **Tour-pool merging** — every 10 kicks, two elite tours from the pool
       are merged to create a new starting point.
    6. **Best-solution tracking** — lexicographic (penalty, cost) objective.

    Args:
        distance_matrix: (n × n) symmetric cost matrix.
        initial_tour: Optional starting tour (closed).  Nearest-neighbour
            construction is used when not provided.
        waste: 1-D demand array for VRP.
        capacity: Vehicle capacity constraint.
        revenue: Revenue per unit collected (VRPP parameter).
        cost_unit: Cost per distance unit.
        mandatory_nodes: Nodes that must be visited.
        coords: Node coordinates (N×2) for geometric operators.
        max_trials: Maximum number of kicks / restarts.
        popmusic_subpath_size: POPMUSIC sub-path size.
        popmusic_trials: Number of POPMUSIC runs.
        popmusic_max_candidates: Maximum number of candidates for POPMUSIC.
        max_k_opt: Maximum k for k-opt moves (2-5).
        use_ip_merging: If True, use IP-based tour recombination.
        max_pool_size: Maximum number of elite solutions to retain.
        profit_aware_operators: Toggle VRPP mode (subset selection).
        alns_iterations: Number of ALNS iterations (VRPP mode only).
        plateau_limit: Iterations before destroy-repair.
        deep_plateau_limit: Iterations before perturbation.
        perturb_operator_weights: Weights for perturbation operators.
        n_vehicles: Number of vehicles.
        n_original: Original graph size before augmentation (for penalty
            calculation with augmented dummy depots).  If None, equals ``n``.
        candidate_set: Pre-computed candidate neighbour lists.  When provided,
            the expensive MST/POPMUSIC candidate generation is skipped
            entirely, enabling warm-start across LNS iterations.
        recorder: Optional recorder for visualisation / diagnostics.
        np_rng: Numpy random generator; seeded from 42 if not provided.
        rng: Random number generator forwarded to operators.
        seed: Seed for the random number generator.
        dynamic_topology_discovery: Whether to discover k-opt topologies dynamically.
        native_prize_collecting: Whether to use native prize collecting algorithm.
        subgradient_iterations: Number of subgradient optimization iterations.
        sa_max_trials: Number of simulated annealing trials.

    Returns:
        Tuple[List[List[int]], float, float]: Best routes, travel cost, and penalty.
    """
    n = len(distance_matrix)

    # Degeneracy check: zero off-diagonal distances cause infinite loops
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        if not np.all(distance_matrix[mask] > 0):
            warnings.warn(
                "Zero off-diagonal distances detected in the distance matrix. "
                "Co-located nodes may cause infinite loops in LKH-3 k-opt search.",
                RuntimeWarning,
                stacklevel=2,
            )

    if n < 3:
        tour = list(range(n)) + [0]
        cost = sum(distance_matrix[tour[i], tour[i + 1]] for i in range(n))
        return [tour], float(cost), 0.0

    if np_rng is None:
        np_rng = np.random.default_rng(seed)

    # Bridge numpy RNG (used for array ops) → stdlib Random (operator interfaces)
    stdlib_rng = Random(seed) if rng is None else rng

    # --- Phase 3: Multi-vehicle Graph Augmentation ---
    # Dummy depots must be added BEFORE initialization or alpha-measure calculation.
    # We only augment if n_vehicles > 1 (single vehicle TSP/CVRP uses original graph).
    if n_vehicles > 1:
        # Convert waste array back to dict node_id: value for augment_graph
        waste_dict = {i: float(waste[i]) for i in range(1, len(waste))} if waste is not None else {}
        distance_matrix, waste, n_original = augment_graph(
            distance_matrix,
            waste_dict,
            n_vehicles=n_vehicles,
            capacity=capacity,
        )
    else:
        # Single vehicle or TSP
        if n_original <= 0:
            n_original = n

    # 1. Initialisation
    curr_tour = _initialize_tour(distance_matrix, initial_tour)

    # 2. Candidate sets — reuse cached if provided, else generate fresh
    if candidate_set is not None:
        candidates = candidate_set
    elif n > 1000:
        candidates = popmusic_candidates(
            distance_matrix,
            curr_tour,
            subpath_size=popmusic_subpath_size,
            n_runs=popmusic_trials,
            max_candidates=popmusic_max_candidates,
            np_rng=np_rng,
        )
    else:
        # Held-Karp subgradient optimization for pi-penalties
        pi = None
        if subgradient_iterations > 0:
            pi = solve_subgradient(distance_matrix, max_iterations=subgradient_iterations, n_original=n_original)

        alpha = compute_alpha_measures(distance_matrix, pi=pi)
        candidates = get_candidate_set(distance_matrix, alpha, max_candidates=popmusic_max_candidates)

    # --- Phase 3: Candidate Set Validation (Multi-vehicle) ---
    # If using a provided candidate set for an augmented graph, we must ensure:
    # 1. Dummy depots have entries (copied from main depot 0)
    # 2. Other nodes include dummy depots in their candidate lists if they include 0
    if candidates is not None and n_vehicles > 1:
        dummy_indices = list(range(n_original, len(distance_matrix)))
        depot_cands = candidates.get(0, [])
        new_candidates = candidates.copy()

        # 1. Add entries for dummies themselves
        for i in dummy_indices:
            if i not in new_candidates:
                new_candidates[i] = depot_cands[:]

        # 2. Reciprocal candidates: if node -> 0 is possible, node -> dummy is possible
        for node, cands in new_candidates.items():
            if 0 in cands:
                # Add dummy indices to candidates list (avoid duplicates)
                filtered_dummies = [d for d in dummy_indices if d not in cands]
                if filtered_dummies:
                    new_candidates[node] = cands + filtered_dummies

        candidates = new_candidates

    curr_pen, curr_cost = get_score(curr_tour, distance_matrix, waste, capacity, n_original)
    best_tour = curr_tour[:]
    best_pen, best_cost = curr_pen, curr_cost
    lkh_criterion = None
    if sa_max_trials > 0:
        # Start temp configured to accept ~5% worse initial routing cost with 50% probability
        start_temp = (curr_cost * 0.05) / 0.693 if curr_cost > 0 else 10.0
        lkh_criterion = AcceptanceCriterionFactory.create(
            name="bmc",
            initial_temp=start_temp,
            alpha=0.999,  # Very slow decay for deep k-opt exploration
            seed=seed,
        )

    dont_look_bits: Optional[np.ndarray] = None

    # Tour pool for elite-solution recombination
    tour_pool: List[List[int]] = [best_tour[:]]

    # 3. Main iterated local-search loop
    for restart in range(max_trials):
        # Inner local-search until no improving k-opt move exists
        while True:
            curr_tour, curr_pen, curr_cost, improved_local, dont_look_bits = _improve_tour(
                curr_tour,
                curr_pen,
                curr_cost,
                candidates,
                distance_matrix,
                waste,
                capacity,
                stdlib_rng,
                dont_look_bits,
                max_k_opt,
                n_original,
                dynamic_topology_discovery=dynamic_topology_discovery,
                acceptance_criterion=lkh_criterion,  # THREAD THE CRITERION
            )
            if not improved_local:
                break

        # Update global best
        if is_better(curr_pen, curr_cost, best_pen, best_cost):
            best_tour = curr_tour[:]
            best_pen, best_cost = curr_pen, curr_cost
            tour_pool.append(best_tour[:])
            if len(tour_pool) > max_pool_size:
                tour_pool.pop(0)

        if lkh_criterion is not None:
            lkh_criterion.step(current_obj=-best_cost, candidate_obj=-best_cost, accepted=True)

        if recorder is not None:
            recorder.record(
                iteration=restart,
                best_obj=-best_cost,
                curr_cost=curr_cost,
                best_penalty=best_pen,
            )

        # Every 10 restarts: try tour-pool merging
        if restart > 0 and restart % 10 == 0 and len(tour_pool) >= 2:
            merged_tour = merge_tours_best(tour_pool, distance_matrix, use_ip=use_ip_merging)
            merged_pen, merged_cost = get_score(merged_tour, distance_matrix, waste, capacity, n_original)

            if is_better(merged_pen, merged_cost, best_pen, best_cost):
                best_tour = merged_tour[:]
                best_pen, best_cost = merged_pen, merged_cost
                tour_pool.append(merged_tour[:])
                if len(tour_pool) > max_pool_size:
                    tour_pool.pop(0)

            curr_tour = merged_tour
            curr_pen, curr_cost = merged_pen, merged_cost
            dont_look_bits = None
        else:
            # Double-bridge perturbation via the shared operator
            curr_tour = _double_bridge_kick(best_tour, distance_matrix, stdlib_rng)
            curr_pen, curr_cost = get_score(curr_tour, distance_matrix, waste, capacity, n_original)
            dont_look_bits = None

    # Extract routes from tour
    routes = decode_augmented_tour(best_tour, n_original)
    return routes, best_cost, best_pen


# ---------------------------------------------------------------------------
# ALNS-Integrated Solver (Phase 2)
# ---------------------------------------------------------------------------


__all__ = ["solve_lkh3"]
