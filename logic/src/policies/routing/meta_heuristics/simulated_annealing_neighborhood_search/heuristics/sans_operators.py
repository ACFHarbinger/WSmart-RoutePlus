"""
Operator application logic for Simulated Annealing.

Provides routines for applying neighborhood operators and perturbations to
solutions within a Simulated Annealing framework. Supported operations include
2-opt, moves between routes, swaps, insertions, removals, and various
relocation, addition, and route-level modifications.

Attributes:
    None

Example:
    None
"""

from random import Random
from typing import Any, Dict, List, Set

import numpy as np

from .sans_neighborhoods import (
    cross_exchange,
    get_2opt_neighbors,
    insert_bin_in_route,
    move_between_routes,
    mutate_route_by_swapping_bins,
    or_opt_move,
    relocate_within_route,
)
from .sans_perturbations import (
    add_n_bins_consecutive,
    add_n_bins_random,
    add_route_with_removed_bins_consecutive,
    add_route_with_removed_bins_random,
    move_n_route_consecutive,
    move_n_route_random,
    remove_bins_from_route,
    remove_n_bins_consecutive,
    remove_n_bins_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)


def _handle_2opt(
    solution: List[List[int]],
    rng: Random,
) -> List[List[int]]:
    """Apply 2-opt operator.

    Args:
        solution: The solution to apply the operator to.
        rng: Random number generator.

    Returns:
        The solution after applying the operator.
    """
    valid_indices = [i for i, r in enumerate(solution) if len(r) > 3]
    if valid_indices:
        r = rng.choice(valid_indices)
        solution[r] = rng.choice(get_2opt_neighbors(solution[r]))
    return solution


def _handle_move(
    solution: List[List[int]],
    data: Dict[str, Any],
    vehicle_capacity: float,
    id_to_index: Dict[int, int],
    rng: Random,
) -> List[List[int]]:
    """Apply move operator.

    Args:
        solution: The solution to apply the operator to.
        data: Problem data.
        vehicle_capacity: Capacity of vehicles.
        id_to_index: Mapping from bin IDs to indices.
        rng: Random number generator.

    Returns:
        The solution after applying the operator.
    """
    neighbors = move_between_routes(solution, data, vehicle_capacity, id_to_index, rng)
    if neighbors:
        return rng.choice(neighbors)
    return solution


def _handle_swap(solution, rng):
    """Apply swap operator.

    Args:
        solution: The solution to apply the operator to.
        rng: Random number generator.

    Returns:
        The solution after applying the operator.
    """
    r = rng.choice(range(len(solution)))
    solution[r] = mutate_route_by_swapping_bins(solution[r], rng, num_bins=rng.choice([1, 2]))
    return solution


def _handle_insert(
    solution: List[List[int]],
    data: Dict[str, Any],
    stocks: Dict[int, float],
    vehicle_capacity: float,
    id_to_index: Dict[int, int],
    distance_matrix: np.ndarray,
    candidate_removed_bins: Set[int],
    rng: Random,
) -> List[List[int]]:
    """Apply insert operator.

    Args:
        solution: The solution to apply the operator to.
        data: Problem data.
        stocks: Stock levels for each bin.
        vehicle_capacity: Capacity of vehicles.
        id_to_index: Mapping from bin IDs to indices.
        distance_matrix: Distance matrix for routes.
        candidate_removed_bins: Set of bins that can be removed.
        rng: Random number generator.

    Returns:
        The solution after applying the operator.
    """
    if not solution:
        return solution
    r = rng.choice(range(len(solution)))
    all_bins = set(data["#bin"]) - {0}
    used_bins = set(b for route in solution for b in route)
    unused = sorted(list(all_bins - used_bins))
    if unused:
        bin_to_insert = rng.choice(unused)
        load = sum(stocks.get(b, 0) for b in solution[r] if b != 0)
        if load + stocks.get(bin_to_insert, 0) <= vehicle_capacity:
            solution[r] = insert_bin_in_route(solution[r], bin_to_insert, id_to_index, distance_matrix)
            if bin_to_insert in candidate_removed_bins:
                candidate_removed_bins.remove(bin_to_insert)
    return solution


def apply_operator(
    op: str,
    new_solution: List[List[int]],
    candidate_removed_bins: Set[int],
    data: dict,
    vehicle_capacity: float,
    id_to_index: dict,
    stocks: Dict[int, float],
    mandatory_bins: Set[int],
    distance_matrix: np.ndarray,
    rng: Random,
) -> List[List[int]]:
    """
    Apply the selected operator to the solution.
    Modifies new_solution in place or returns a new one.

    Args:
        op: The operator to apply (string).
        new_solution: The solution to apply the operator to.
        candidate_removed_bins: Set of bins that can be removed.
        data: Problem data.
        vehicle_capacity: Capacity of vehicles.
        id_to_index: Mapping from bin IDs to indices.
        stocks: Stock levels for each bin.
        mandatory_bins: Set of bins that cannot be removed.
        distance_matrix: Distance matrix for routes.
        rng: Random number generator.

    Returns:
        The solution after applying the operator.
    """
    # Simple operators using direct function calls or simple lambdas
    handlers = {
        "2opt": lambda sol: _handle_2opt(sol, rng),
        "move": lambda sol: _handle_move(sol, data, vehicle_capacity, id_to_index, rng),
        "swap": lambda sol: _handle_swap(sol, rng),
        "remove": lambda sol: remove_bins_from_route(sol, mandatory_bins, rng, num_bins=1),
        "move_n_random": lambda sol: move_n_route_random(sol, rng, n=rng.randint(2, 5)),
        "move_n_consec": lambda sol: move_n_route_consecutive(sol, rng, n=rng.randint(2, 5)),
        "swap_n_random": lambda sol: swap_n_route_random(sol, rng, n=rng.randint(2, 5)),
        "swap_n_consec": lambda sol: swap_n_route_consecutive(sol, rng, n=rng.randint(2, 5)),
        "remove_n_bins": lambda sol: remove_n_bins_random(
            sol, candidate_removed_bins, mandatory_bins, rng, n=rng.randint(2, 5)
        ),
        "remove_n_bins_consec": lambda sol: remove_n_bins_consecutive(
            sol, candidate_removed_bins, mandatory_bins, rng, n=rng.randint(2, 5)
        ),
        "add_n_bins": lambda sol: add_n_bins_random(
            sol, candidate_removed_bins, stocks, vehicle_capacity, id_to_index, distance_matrix, rng, n=2
        ),
        "add_n_bins_consec": lambda sol: add_n_bins_consecutive(
            sol, candidate_removed_bins, stocks, vehicle_capacity, id_to_index, distance_matrix, rng, n=2
        ),
        "add_route_removed": lambda sol: add_route_with_removed_bins_random(
            sol, candidate_removed_bins, stocks, vehicle_capacity, rng
        ),
        "add_route_removed_consec": lambda sol: add_route_with_removed_bins_consecutive(
            sol, candidate_removed_bins, stocks, vehicle_capacity, rng
        ),
        "relocate": lambda sol: (
            sol.__setitem__(idx := rng.choice(range(len(sol))), relocate_within_route(sol[idx], rng)) or sol
        ),
        "cross": lambda sol: cross_exchange(sol, rng),
        "or-opt": lambda sol: sol.__setitem__(idx := rng.choice(range(len(sol))), or_opt_move(sol[idx], rng)) or sol,
        "insert": lambda sol: _handle_insert(
            sol, data, stocks, vehicle_capacity, id_to_index, distance_matrix, candidate_removed_bins, rng
        ),
    }

    handler = handlers.get(op)
    if handler:
        return handler(new_solution)
    return new_solution
