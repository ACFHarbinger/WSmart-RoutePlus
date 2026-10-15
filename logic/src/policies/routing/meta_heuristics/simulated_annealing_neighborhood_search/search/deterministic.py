"""
Deterministic local search strategies.

Attributes:
    _evaluate_move: Evaluate moving a bin to a new position.
    local_search_2: Apply a deterministic set of local search operators to improve a solution.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.search.deterministic import local_search_2
    >>> routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> distance_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    >>> values = np.array([10, 10, 10])
    >>> data = {'nodes': 5}
    >>> local_search_2(routes, 1, 1, 1, 1, data, distance_matrix, values)
    ([0, 1, 2, 0], [0, 3, 4, 0]), 0, 0
"""

from copy import deepcopy
from typing import List, Tuple

import numpy as np

from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.objectives import (
    compute_profit,
    compute_real_profit,
)


def _evaluate_move(
    routes_list: List[List[int]],
    idx_route: int,
    position: int,
    bin_to_move: int,
    previous_solution: List[List[int]],
    previous_profit: float,
    p_vehicle: float,
    p_load: float,
    p_route_difference: float,
    p_shift: float,
    data: dict,
    distance_matrix: np.ndarray,
    values: np.ndarray,
) -> Tuple[float, bool]:
    """Evaluate moving a bin to a new position.

    Args:
        routes_list: Current routing solution.
        idx_route: Index of the route to move the bin to.
        position: Position to move the bin to.
        bin_to_move: Bin to move.
        previous_solution: Previous routing solution.
        previous_profit: Previous profit.
        p_vehicle: Penalty for using an extra vehicle.
        p_load: Penalty for exceeding vehicle load capacity.
        p_route_difference: Penalty for route length differences.
        p_shift: Penalty for route time differences.
        data: Input data containing bin information.
        distance_matrix: All-pairs shortest path distances.
        values: Vehicle capacity.

    Returns:
        Tuple[float, bool]: New profit and whether the move improved the solution.
    """
    index_route_to_remove = -1
    for z_idx, z in enumerate(previous_solution):
        if bin_to_move in z:
            index_route_to_remove = z_idx
            break

    if index_route_to_remove == -1:
        return previous_profit, False

    # Determine insertion place
    if index_route_to_remove == idx_route:
        current_route = previous_solution[idx_route]
        position_bin_to_move = current_route.index(bin_to_move)
        place = position if position_bin_to_move < position else position + 1
    else:
        place = position + 1

    # Temporarily apply move
    if index_route_to_remove == idx_route:
        routes_list[idx_route].remove(bin_to_move)
        routes_list[idx_route].insert(place, bin_to_move)
    else:
        routes_list[index_route_to_remove].remove(bin_to_move)
        routes_list[idx_route].insert(place, bin_to_move)

    new_profit = compute_profit(
        routes_list,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    if new_profit > previous_profit:
        return new_profit, True

    return previous_profit, False


def local_search_2(
    previous_solution: List[List[int]],
    p_vehicle: float,
    p_load: float,
    p_route_difference: float,
    p_shift: float,
    data: dict,
    distance_matrix: np.ndarray,
    values: np.ndarray,
) -> Tuple[List[List[int]], float, float]:
    """
    Apply a deterministic set of local search operators to improve a solution.

    Args:
        previous_solution: Current routing solution.
        p_vehicle: Penalty for using an extra vehicle.
        p_load: Penalty for exceeding vehicle load capacity.
        p_route_difference: Penalty for route length differences.
        p_shift: Penalty for route time differences.
        data: Input data containing bin information.
        distance_matrix: All-pairs shortest path distances.
        values: Vehicle capacity.

    Returns:
        Tuple[List[List[int]], float, float]: Optimized routing solution, its profit, and real profit.
    """
    previous_profit = compute_profit(
        previous_solution,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )
    routes_list = deepcopy(previous_solution)

    # Loop over routes
    for idx_route, _route in enumerate(previous_solution):
        # iterate current route bins (except last depot)
        current_route_data = previous_solution[idx_route]
        for val in current_route_data[:-1]:  # skip last depot
            position = current_route_data.index(val)
            row = list(distance_matrix[val][:])
            row_new = sorted(row)

            # Check closest bins (excluding self at 0 distance)
            candidates = row_new[1:51]
            candidates = []
            count = 0
            for dist in row_new[1:]:
                b_idx = row.index(dist)
                if b_idx != 0:
                    candidates.append(b_idx)
                    count += 1
                if count >= 50:
                    break

            # Now we have candidates.
            # Original logic applied move for each candidate.
            for bin_to_move in candidates:
                new_profit, improved = _evaluate_move(
                    routes_list,
                    idx_route,
                    position,
                    bin_to_move,
                    previous_solution,
                    previous_profit,
                    p_vehicle,
                    p_load,
                    p_route_difference,
                    p_shift,
                    data,
                    distance_matrix,
                    values,
                )

                if improved:
                    previous_solution = deepcopy(routes_list)
                    previous_profit = new_profit
                else:
                    # Revert
                    routes_list = deepcopy(previous_solution)

    solution_after_local_search = deepcopy(previous_solution)
    profit_after_local_search = compute_profit(
        solution_after_local_search,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )
    LS_profit = compute_real_profit(solution_after_local_search, p_vehicle, data, distance_matrix, values)
    return solution_after_local_search, profit_after_local_search, LS_profit
