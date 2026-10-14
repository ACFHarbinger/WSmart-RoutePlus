"""
Simulated Annealing Algorithm.

Attributes:
    improved_simulated_annealing: Refine routing solutions using a multi-neighborhood Simulated Annealing algorithm.

Example:
    >>> import random
    >>> routes = [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]]
    >>> bins_cannot_removed = {1, 4}
    >>> rng = random.Random()
    >>> removed_bins = set()
    >>> new_routes = remove_bins_from_route(routes, bins_cannot_removed, rng, num_bins=1)
    >>> new_routes
    [[0, 2, 3, 0], [0, 4, 5, 6, 0]]
"""

import copy
import math
import random
import time
from random import Random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.routes import (
    uncross_arcs_in_sans_routes,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_operators import (
    apply_operator,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state import (
    compute_profit,
)
from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _initialize_solution_state(
    routes: List[List[int]],
    id_to_index: Dict[int, int],
    distance_matrix: np.ndarray,
    data: Dict[str, Any],
) -> Tuple[List[List[int]], Set[int]]:
    """Initialize SANS solution state.

    Args:
        routes: Initial routes to optimize.
        id_to_index: Mapping from bin IDs to indices.
        distance_matrix: Distance matrix for routes.
        data: Problem data.

    Returns:
        Tuple: (current_solution, missing_bins).

    Notes:
        - State Sync: Captures any bins missed by the initial greedy solution.
        - Every bin ID must be either in a route or in the 'removed_bins' set.
        - Route improvement: Uncross arcs in the initial solution to start from a geometric optimum.
    """
    all_bins_ids = set(data["#bin"].tolist()) - {0}
    scheduled_bins = set()
    for r in routes:
        for b in r:
            if b != 0:
                scheduled_bins.add(b)
    missing_bins = all_bins_ids - scheduled_bins

    initial_solution = copy.deepcopy(routes)
    # Route improvement: Uncross arcs in the initial solution to start from a geometric optimum.
    current_solution = uncross_arcs_in_sans_routes(initial_solution, id_to_index, distance_matrix)

    return current_solution, missing_bins


def _select_neighbor(
    solution: List[List[int]],
    removed_bins: Set[int],
    data: Dict[str, Any],
    vehicle_capacity: float,
    id_to_index: Dict[int, int],
    stocks: Dict[int, float],
    mandatory_bins: Set[int],
    distance_matrix: np.ndarray,
    rng: Random,
) -> Tuple[Optional[List[List[int]]], Optional[str]]:
    """Select and apply a neighbor operator.

    Args:
        solution: The current solution.
        removed_bins: Set of bins that have been removed.
        data: Problem data.
        vehicle_capacity: Capacity of vehicles.
        id_to_index: Mapping from bin IDs to indices.
        stocks: Stock levels for each bin.
        mandatory_bins: Set of bins that must be included in the solution.
        distance_matrix: Distance matrix for routes.
        rng: Random number generator.

    Returns:
        A tuple containing the new solution and the operator applied.
    """
    route_ops = [
        "2opt",
        "move",
        "swap",
        "remove",
        "insert",
        "move_n_random",
        "move_n_consec",
        "swap_n_random",
        "swap_n_consec",
        "remove_n_bins",
        "remove_n_bins_consec",
        "relocate",
        "cross",
        "or-opt",
    ]
    add_ops = [
        "add_n_bins",
        "add_n_bins_consec",
        "add_route_removed",
        "add_route_removed_consec",
    ]

    valid_ops = []
    if len(solution) > 0:
        valid_ops.extend(route_ops)
    if len(removed_bins) > 0:
        valid_ops.extend(add_ops)

    if not valid_ops:
        return None, None

    op = rng.choice(valid_ops)

    new_solution = apply_operator(
        op,
        copy.deepcopy(solution),
        copy.deepcopy(removed_bins),
        data,
        vehicle_capacity,
        id_to_index,
        stocks,
        mandatory_bins,
        distance_matrix,
        rng=rng,
    )

    # Filter out empty routes
    new_solution = [r for r in new_solution if len(r) > 2]

    return new_solution, op


def improved_simulated_annealing(  # noqa: C901
    routes,
    distance_matrix,
    time_limit,
    id_to_index,
    data,
    vehicle_capacity,
    T_init=1000,
    T_min=0.001,
    alpha=0.995,
    iterations_per_T=100,
    R=0.165,
    V=2.5,
    density=20,
    C=1.0,
    mandatory_bins=None,
    removed_bins=None,
    verbose=False,
    perc_bins_can_overflow=0.0,
    volume=2.5,
    density_val=20,
    max_vehicles=None,
    recorder: Optional[PolicyStateRecorder] = None,
    rng: Optional[random.Random] = None,
):
    """Refine routing solutions using a multi-neighborhood Simulated Annealing algorithm.

    Args:
        routes (List[List[int]]): Initial routes to optimize.
        distance_matrix (np.ndarray): Symmetric distance matrix.
        time_limit (float): Maximum allowed runtime in seconds.
        id_to_index (Dict[int, int]): Mapping of node IDs to matrix indices.
        data (pd.DataFrame): Dataframe containing bin metadata (ID, Stock).
        vehicle_capacity (float): Maximum vehicle collection capacity.
        T_init (float): Initial temperature for annealing.
        T_min (float): Minimum temperature to stop annealing.
        alpha (float): Cooling rate (T = T * alpha).
        iterations_per_T (int): Number of neighbor moves at each temperature step.
        R (float): Revenue per kg of waste.
        V (float): Bin volume.
        density (float): Waste density.
        C (float): Cost per km traveled.
        mandatory_bins (Optional[Set[int]]): Bins that must be collected.
        removed_bins (Optional[Set[int]]): Bins initially not in any route.
        verbose (bool): Whether to print debug information.
        perc_bins_can_overflow (float): Allowed overflow percentage.
        volume (float): Bin volume (redundant with V).
        density_val (float): Waste density (redundant with density).
        max_vehicles (Optional[int]): Maximum number of vehicles allowed.
        recorder (Optional[PolicyStateRecorder]): Telemetry recorder.
        rng (Optional[random.Random]): Random number generator.

    Returns:
        Tuple: (best_solution, best_profit, last_distance, last_weight, last_receita).
    """
    if rng is None:
        rng = random.Random()

    start_time = time.perf_counter()

    # --- 1. ROBUST INITIALIZATION ---
    removed_bins = set() if removed_bins is None else set(removed_bins)
    mandatory_bins = set() if mandatory_bins is None else set(mandatory_bins)

    current_solution, missing_bins = _initialize_solution_state(routes, id_to_index, distance_matrix, data)
    removed_bins.update(missing_bins)

    stocks = dict(zip(data["#bin"], data["Stock"], strict=False))

    # Initial Evaluation
    current_profit, current_cost, current_revenue, real_kg = compute_profit(
        current_solution,
        distance_matrix,
        id_to_index,
        data,
        vehicle_capacity,
        R,
        V,
        density,
        mandatory_bins,
        stocks,
    )

    best_solution = copy.deepcopy(current_solution)
    best_profit = current_profit
    last_receita = current_revenue
    last_weight = real_kg
    last_distance = current_cost

    no_improvement_count = 0
    T = T_init

    # --- 2. MAIN SIMULATED ANNEALING LOOP ---
    while T_min < T:
        if time.perf_counter() - start_time > time_limit:
            if verbose:
                print("[DEBUG] Time limit reached.")
            break

        # Inner Loop
        for _ in range(iterations_per_T):
            if time.perf_counter() - start_time > time_limit:
                break

            # --- 3/4. NEIGHBOR SELECTION AND EVALUATION ---
            # (A first _select_neighbor call here produced a neighbour that was always
            # discarded in favour of the operator applied below.)

            candidate_removed_bins = copy.deepcopy(removed_bins)
            route_ops = [
                "2opt",
                "move",
                "swap",
                "remove",
                "insert",
                "move_n_random",
                "move_n_consec",
                "swap_n_random",
                "swap_n_consec",
                "remove_n_bins",
                "remove_n_bins_consec",
                "relocate",
                "cross",
                "or-opt",
            ]
            add_ops = ["add_n_bins", "add_n_bins_consec", "add_route_removed", "add_route_removed_consec"]

            valid_ops = []
            if len(current_solution) > 0:
                valid_ops.extend(route_ops)
            if len(candidate_removed_bins) > 0:
                valid_ops.extend(add_ops)

            if not valid_ops:
                continue

            op = rng.choice(valid_ops)
            new_solution = apply_operator(
                op,
                copy.deepcopy(current_solution),
                candidate_removed_bins,
                data,
                vehicle_capacity,
                id_to_index,
                stocks,
                mandatory_bins,
                distance_matrix,
                rng=rng,
            )
            new_solution = [r for r in new_solution if len(r) > 2]

            new_profit, new_cost, new_revenue, new_real_kg = compute_profit(
                new_solution,
                distance_matrix,
                id_to_index,
                data,
                vehicle_capacity,
                R,
                V,
                density,
                mandatory_bins,
                stocks,
            )

            delta = new_profit - current_profit

            # Acceptance
            accept = False
            if delta > 0:
                accept = True
            else:
                try:
                    p = math.exp(delta / T)
                except OverflowError:
                    p = 0
                accept = rng.random() < p

            if accept:
                current_solution = new_solution
                current_cost = new_cost
                current_profit = new_profit
                removed_bins = candidate_removed_bins

                if current_profit > best_profit:
                    best_solution = copy.deepcopy(current_solution)
                    best_profit = current_profit
                    last_weight = new_real_kg
                    last_distance = new_cost
                    last_receita = new_revenue
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

        T = T * alpha
        if verbose:
            print(f"Temperature cooled to {T:.4f}")

        if recorder is not None:
            recorder.record(
                temperature=T,
                best_profit=best_profit,
                current_profit=current_profit,
                no_improvement_count=no_improvement_count,
            )

        if no_improvement_count > 500:
            if verbose:
                print("\n[INFO] Rehearing temperature.")
            T = T_init
            no_improvement_count = 0

    best_solution = [r for r in best_solution if len(r) > 2]
    best_solution = uncross_arcs_in_sans_routes(best_solution, id_to_index, distance_matrix)
    return best_solution, best_profit, last_distance, last_weight, last_receita
