"""
Solution refinement strategies for the look-ahead policy.
Provides iterative improvement passes using uncrossing and local search variants.

Attributes:
    refine_solution: Apply a sequence of uncrossing and local search passes to refine a solution.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.refinement.refinement import refine_solution
    >>> routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> cost, profit = refine_solution(routes, distance_matrix, values, capacities, mandatory_nodes)
    >>> print(f"Cost: {cost}, Profit: {profit}")
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from logic.src.policies.helpers.operators.improvement_descent.relocate_descent import relocate_descent
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.objectives import (
    compute_profit,
    compute_real_profit,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.routes import (
    uncross_arcs_in_routes,
)


def refine_solution(
    solution: List[List[int]],
    p_vehicle: float,
    p_load: float,
    p_route_difference: float,
    p_shift: float,
    data: pd.DataFrame,
    points: Dict,
    distance_matrix: np.ndarray,
    values: Dict,
    iterations: int = 5,
) -> Tuple[List[List[int]], float]:
    """
    Apply a sequence of uncrossing and local search passes to refine a solution.

    Args:
        solution: Current routing solution.
        p_vehicle: Vehicle penalty.
        p_load: Load penalty.
        p_route_difference: Imbalance penalty.
        p_shift: Shift penalty.
        data: Bin weights data.
        points: Node coordinates.
        distance_matrix: Distance matrix.
        values: Parameter dictionary.
        iterations: Number of refinement loops.

    Returns:
        Tuple of (refined_solution, final_profit).
    """

    def profit_fn(routes):
        return compute_profit(routes, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    def real_profit_fn(routes):
        return compute_real_profit(routes, p_vehicle, data, distance_matrix, values)

    current_sol = solution
    final_profit = 0.0

    for _ in range(iterations):
        # 1. Forward Pass
        current_sol, _, _ = uncross_arcs_in_routes(
            current_sol,
            p_vehicle,
            p_load,
            p_route_difference,
            p_shift,
            data,
            points,
            distance_matrix,
            values,
        )
        current_sol, final_profit, _ = relocate_descent(current_sol, profit_fn, real_profit_fn, distance_matrix)

        # 2. Backward/Reversed Pass
        current_sol, _, _ = uncross_arcs_in_routes(
            current_sol,
            p_vehicle,
            p_load,
            p_route_difference,
            p_shift,
            data,
            points,
            distance_matrix,
            values,
        )
        current_sol, final_profit, _ = relocate_descent(
            current_sol, profit_fn, real_profit_fn, distance_matrix, reverse=True
        )

    return current_sol, final_profit
