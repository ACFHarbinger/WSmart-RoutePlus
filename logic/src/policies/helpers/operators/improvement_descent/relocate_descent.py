"""
Nearest-neighbour relocate descent on depot-terminated routes.

One pass over every bin of every route: each of the bin's 50 nearest bins is
moved next to it (just after it, or just before it when the moved bin sat
earlier in the same route), and the move is kept when it raises the given
objective. This is LAC's deterministic local search (Jorge et al. 2022); the
forward pass scans routes and bins in order, the reversed pass in reverse.

Attributes:
    relocate_descent: One improving pass of nearest-neighbour relocations.

Example:
    >>> from logic.src.policies.helpers.operators.improvement_descent.relocate_descent import relocate_descent
    >>> routes, profit, real_profit = relocate_descent(routes, profit_fn, real_profit_fn, distance_matrix)
"""

from copy import deepcopy
from typing import Callable, List, Tuple

import numpy as np

Objective = Callable[[List[List[int]]], float]


def _evaluate_move(
    routes_list: List[List[int]],
    idx_route: int,
    position: int,
    bin_to_move: int,
    previous_solution: List[List[int]],
    previous_profit: float,
    profit_fn: Objective,
) -> Tuple[float, bool]:
    """Apply moving ``bin_to_move`` next to ``routes_list[idx_route][position]`` and score it.

    Args:
        routes_list: Working copy of the solution; the move is applied to it.
        idx_route: Index of the route to move the bin to.
        position: Position of the anchor bin in that route.
        bin_to_move: Bin to move.
        previous_solution: Solution before the move.
        previous_profit: Objective of ``previous_solution``.
        profit_fn: Objective to maximise.

    Returns:
        Tuple[float, bool]: New objective and whether the move improved the solution.
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

    new_profit = profit_fn(routes_list)

    if new_profit > previous_profit:
        return new_profit, True

    return previous_profit, False


def relocate_descent(
    previous_solution: List[List[int]],
    profit_fn: Objective,
    real_profit_fn: Objective,
    distance_matrix: np.ndarray,
    reverse: bool = False,
) -> Tuple[List[List[int]], float, float]:
    """
    One improving pass of nearest-neighbour relocations over the whole solution.

    Args:
        previous_solution: Current routing solution (depot-terminated routes).
        profit_fn: Objective to maximise, such as the penalised profit.
        real_profit_fn: Objective reported for the result, such as the unpenalised profit.
        distance_matrix: All-pairs shortest path distances.
        reverse: Scan routes, and bins within each route, from last to first.

    Returns:
        Tuple[List[List[int]], float, float]: Improved solution, its ``profit_fn`` and its ``real_profit_fn``.
    """
    previous_profit = profit_fn(previous_solution)
    routes_list = deepcopy(previous_solution)

    route_order = range(len(previous_solution))
    for idx_route in reversed(route_order) if reverse else route_order:
        current_route_data = previous_solution[idx_route]
        bins_to_check = current_route_data[:-1]  # skip last depot
        for val in reversed(bins_to_check) if reverse else bins_to_check:
            position = current_route_data.index(val)
            row = list(distance_matrix[val][:])
            row_new = sorted(row)

            # Closest 50 bins (excluding self at 0 distance and the depot)
            candidates = []
            count = 0
            for dist in row_new[1:]:
                b_idx = row.index(dist)
                if b_idx != 0:
                    candidates.append(b_idx)
                    count += 1
                if count >= 50:
                    break

            for bin_to_move in candidates:
                new_profit, improved = _evaluate_move(
                    routes_list,
                    idx_route,
                    position,
                    bin_to_move,
                    previous_solution,
                    previous_profit,
                    profit_fn,
                )

                if improved:
                    previous_solution = deepcopy(routes_list)
                    previous_profit = new_profit
                else:
                    # Revert
                    routes_list = deepcopy(previous_solution)

    solution = deepcopy(previous_solution)
    return solution, profit_fn(solution), real_profit_fn(solution)
