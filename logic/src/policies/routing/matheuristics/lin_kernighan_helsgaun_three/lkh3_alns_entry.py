"""LKH-3 with the ALNS outer loop (prize-collecting / expanding-pool entry point).

Attributes:
    solve_lkh3_with_alns: Run ``LKH3_ALNS`` over the candidate pool and return the best plan.

Example:
    >>> routes, profit, cost = solve_lkh3_with_alns(dist, wastes, capacity, R, C, params)
"""

from __future__ import annotations

from random import Random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def solve_lkh3_with_alns(
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
    # ALNS parameters
    profit_aware_operators: bool = False,
    alns_iterations: int = 100,
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
    Solve VRP/VRPP using LKH-3 + Adaptive Large Neighborhood Search matheuristic.

    This function provides a unified interface that automatically dispatches to:
    - **VRPP mode** (``profit_aware_operators=True``): ALNS + LKH-3 for subset selection

    Algorithm (VRPP mode):
    1. Initialize with greedy profitable subset
    2. For each ALNS iteration:
       a. Optimize routes with LKH-3 k-opt
       b. If improved, update best and elite pool
       c. If plateau, apply destroy-repair operators
       d. If deep plateau, apply perturbation with elite
    3. Return best routes and objective

    Args:
        distance_matrix: (N×N) symmetric distance matrix.
        initial_tour: Optional starting tour (closed).  Nearest-neighbour
            construction is used when not provided.
        waste: 1-D demand array for VRP.
        capacity: Vehicle capacity constraint.
        revenue: Revenue per unit collected (VRPP parameter).
        cost_unit: Cost per distance unit.
        mandatory_nodes: Nodes that must be visited.
        coords: Node coordinates (N×2) for geometric operators.
        max_trials: Max LKH-3 trials per routing optimization.
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
        recorder: Optional state recorder for visualization.
        np_rng: NumPy random generator.
        rng: Random generator.
        seed: Seed for the random number generator.
        dynamic_topology_discovery: Whether to discover k-opt topologies dynamically.
        native_prize_collecting: Whether to use native prize collecting algorithm.
        subgradient_iterations: Number of subgradient optimization iterations.
        sa_max_trials: Number of SA trials.

    Returns:
        Tuple[List[List[int]], float, float]: Routes (each a list of node IDs
            without depot), objective value, and travel cost.

    Example:
        >>> # VRPP mode
        >>> routes, profit = solve_lkh3_with_alns(
        ...     distance_matrix=dist,
        ...     wastes={1: 10, 2: 20, 3: 30},
        ...     capacity=100,
        ...     revenue=2.0,
        ...     cost_unit=1.0,
        ...     profit_aware_operators=True,
        ...     mandatory_nodes=[1],
        ... )
    """
    # Build wastes dict from numpy array (skip depot at index 0)
    if waste is not None:
        wastes: Dict[int, float] = {i: float(waste[i]) for i in range(1, len(waste))}
    else:
        wastes = {}

    if np_rng is None:
        np_rng = np.random.default_rng(seed)

    if rng is None:
        rng = Random(seed)

    from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search import (
        LKH3_ALNS,
    )

    solver = LKH3_ALNS(
        distance_matrix=distance_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue=revenue,
        cost_unit=cost_unit,
        profit_aware_operators=profit_aware_operators,
        mandatory_nodes=mandatory_nodes,
        coords=coords,
        np_rng=np_rng,
        rng=rng,
        seed=seed,
        recorder=recorder,
        max_pool_size=max_pool_size,
        n_original=n_original,
        R=revenue,
        C=cost_unit,
        perturb_operator_weights=perturb_operator_weights,
    )

    routes, objective, penalty = solver.solve(
        max_iterations=alns_iterations,
        lkh_trials=max_trials,
        n_vehicles=n_vehicles,
        plateau_limit=plateau_limit,
        deep_plateau_limit=deep_plateau_limit,
        popmusic_subpath_size=popmusic_subpath_size,
        popmusic_trials=popmusic_trials,
        popmusic_max_candidates=popmusic_max_candidates,
        max_k_opt=max_k_opt,
        use_ip_merging=use_ip_merging,
        subgradient_iterations=subgradient_iterations,
        dynamic_topology_discovery=dynamic_topology_discovery,
        native_prize_collecting=native_prize_collecting,
        sa_max_trials=sa_max_trials,
    )

    return routes, objective, penalty
