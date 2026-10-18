"""
Solution rebalancing strategies for the look-ahead policy.
Iteratively removes and re-inserts bins to optimize route density and feasibility.

Attributes:
    rebalance_solution: Iteratively remove and re-insert bins to balance the solution.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.refinement.rebalancing import rebalance_solution
    >>> routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> cost, profit = rebalance_solution(routes, [], 1.0, 1.0, 1.0, 1.0, data, mandatory_nodes, distance_matrix, values)
    >>> print(f"Cost: {cost}, Profit: {profit}")
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from logic.src.policies.helpers.operators.destroy_ruin.bin_pool import remove_bins_end
from logic.src.policies.helpers.operators.recreate_repair.bin_pool import insert_bins
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.objectives import (
    compute_profit,
    compute_real_profit,
)


def rebalance_solution(
    solution: List[List[int]],
    removed_bins: List[int],
    p_vehicle: float,
    p_load: float,
    p_route_difference: float,
    p_shift: float,
    data: pd.DataFrame,
    mandatory_bins: List[int],
    distance_matrix: np.ndarray,
    values: Dict,
    iterations: int = 10,
) -> Tuple[List[List[int]], float, List[int]]:
    """
    Iteratively remove and re-insert bins to balance the solution.

    Args:
        solution: Current routing solution.
        removed_bins: List of bins currently excluded.
        p_vehicle: Vehicle penalty.
        p_load: Load penalty.
        p_route_difference: Imbalance penalty.
        p_shift: Shift penalty.
        data: Bin data.
        mandatory_bins: Mandatory nodes.
        distance_matrix: Distances.
        values: parameters.
        iterations: Number of swap loops.

    Returns:
        Tuple of (balanced_solution, final_profit, final_removed_bins).
    """

    def profit_fn(routes):
        return compute_profit(routes, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    def real_profit_fn(routes):
        return compute_real_profit(routes, p_vehicle, data, distance_matrix, values)

    current_sol = solution
    current_removed = removed_bins
    final_profit = 0.0

    for _ in range(iterations):
        # 1. Remove Bins from ends/bad positions
        current_sol, _, _, current_removed = remove_bins_end(
            current_sol, current_removed, mandatory_bins, profit_fn, real_profit_fn
        )

        # 2. Re-insert potentially better combinations
        current_sol, _, final_profit, current_removed = insert_bins(
            current_sol, current_removed, profit_fn, real_profit_fn
        )

    return current_sol, final_profit, current_removed
