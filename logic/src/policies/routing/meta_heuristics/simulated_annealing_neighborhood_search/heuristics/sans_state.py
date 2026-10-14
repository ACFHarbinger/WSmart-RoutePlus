"""
State and Cost utilities for Simulated Annealing.

Attributes:
    compute_profit: Calculate the profit of a solution.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.heuristics.sans_state import compute_profit
    >>> solution = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> distance_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    >>> data = {"#bin": [1, 2, 3, 4], "Stock": [10, 20, 30, 40]}
    >>> vehicle_capacity = 100
    >>> R = 1
    >>> V = 1
    >>> density = 1
    >>> mandatory_bins = {1, 2}
    >>> compute_profit(solution, distance_matrix, {1: 1, 2: 2, 3: 3, 4: 4}, data, vehicle_capacity, R, V, density, mandatory_bins)
    (11.0, 3, 14.0, 14.0)
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from logic.src.constants import (
    MAX_CAPACITY_PERCENT,
    PENALTY_MANDATORY_NODES_MISSED,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.distance import (
    compute_total_cost,
)


def compute_profit(
    solution: List[List[int]],
    distance_matrix: List[List[float]],
    id_to_index: Dict[int, int],
    data: Dict[str, Any],
    vehicle_capacity: float,
    R: float,
    V: float,
    density: float,
    mandatory_bins: Set[int],
    stocks: Optional[Dict[int, float]] = None,
) -> Tuple[float, float, float, float]:
    """
    Calculate the profit of a solution.
    Profit = Revenue - Cost - Penalty

    Args:
        solution (List[List[int]]): The current routing solution.
        distance_matrix (List[List[float]]): The distance matrix.
        id_to_index (Dict[int, int]): A mapping from bin IDs to their indices.
        data (Dict[str, Any]): The data containing bin information.
        vehicle_capacity (float): The capacity of the vehicle.
        R (float): The revenue per unit of weight.
        V (float): The volume of a unit of weight.
        density (float): The density of the bins.
        mandatory_bins (Optional[Set[int]]): A set of mandatory bins.
        stocks (Optional[Dict[int, float]]): A dictionary mapping bin IDs to their stock.

    Returns:
        Tuple[float, float, float, float]: The profit, cost, revenue, and real kg.
    """
    if stocks is None:
        stocks = dict(zip(data["#bin"], data["Stock"], strict=False))

    current_cost = compute_total_cost(solution, distance_matrix, id_to_index)

    # Calculate STRICT SIMULATOR PROFIT (Route 0 Only + Capacity Cutoff)
    real_kg = 0
    collected_mandatory = set()
    if len(solution) > 0 and len(solution[0]) > 2:
        route0 = solution[0]
        current_load = 0
        for b in route0:
            if b == 0:
                continue
            bin_kg = stocks.get(b, 0) * V * density / MAX_CAPACITY_PERCENT

            if current_load + bin_kg <= vehicle_capacity:
                current_load += bin_kg
                real_kg += bin_kg
                if mandatory_bins and b in mandatory_bins:
                    collected_mandatory.add(b)
            else:
                break

    current_revenue = real_kg * R

    # Mandatory Penalty: Force all mandatory_bins into the valid trunk of Route 0
    missed_mandatory = len(mandatory_bins) - len(collected_mandatory) if mandatory_bins else 0
    penalty_mandatory = missed_mandatory * PENALTY_MANDATORY_NODES_MISSED

    current_profit = current_revenue - current_cost - penalty_mandatory

    return current_profit, current_cost, current_revenue, real_kg
