"""
Objective functions for SANS.

Attributes:
    compute_profit: Calculate the total objective value (Profit - Penalties).
    compute_real_profit: Calculate the "real" profit (Revenue - Travel Cost - Vehicle penalty).
    compute_total_profit: Calculate the total profit considering revenue and cost.

Example:
    >>> routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> cost, profit = compute_profit(
    ...     routes,
    ...     p_vehicle=0.1,
    ...     p_load=0.1,
    ...     p_route_difference=0.1,
    ...     p_shift=0.1,
    ...     data=data,
    ...     distance_matrix=distance_matrix,
    ...     values=values,
    ... )
    >>> print(f"Cost: {cost}, Profit: {profit}")
"""

from logic.src.constants import MAX_CAPACITY_PERCENT

from .distance import compute_distance_per_route, compute_sans_route_cost
from .penalties import (
    compute_load_excess_penalty,
    compute_route_time_difference_penalty,
    compute_shift_excess_penalty,
    compute_transportation_cost,
    compute_vehicle_use_penalty,
)
from .revenue import compute_waste_collection_revenue


def compute_profit(
    routes_list,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    distance_matrix,
    values,
):
    """
    Calculate the total objective value (Profit - Penalties).

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_vehicle (float): Vehicle penalty coefficient.
        p_load (float): Load penalty coefficient.
        p_route_difference (float): Imbalance penalty coefficient.
        p_shift (float): Shift penalty coefficient.
        data (pd.DataFrame): Bin data.
        distance_matrix (np.ndarray): Distance matrix.
        values (Dict): Parameter dictionary containing keys like E, B, R, C,
            vehicle_capacity, and shift_duration.

    Returns:
        float: Total calculated profit.
    """
    distance_route_vector = compute_distance_per_route(routes_list, distance_matrix)
    total_revenue = compute_waste_collection_revenue(routes_list, data, values["E"], values["B"], values["R"])
    transportation_cost = compute_transportation_cost(routes_list, distance_route_vector, values["C"])
    vehicle_penalty = compute_vehicle_use_penalty(routes_list, p_vehicle)
    route_difference_penalty = compute_route_time_difference_penalty(
        routes_list, p_route_difference, distance_route_vector
    )
    load_excess_penalty = compute_load_excess_penalty(routes_list, p_load, data, values["vehicle_capacity"])
    shift_excess_penalty = compute_shift_excess_penalty(
        routes_list,
        p_shift,
        distance_matrix,
        distance_route_vector,
        values["shift_duration"],
    )

    profit = (
        total_revenue
        - transportation_cost
        - vehicle_penalty
        - route_difference_penalty
        - load_excess_penalty
        - shift_excess_penalty
    )
    return profit


def compute_real_profit(routes_list, p_vehicle, data, distance_matrix, values):
    """
    Calculate the "real" profit (Revenue - Travel Cost - Vehicle penalty)
    excluding soft constraints.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_vehicle (float): Vehicle penalty.
        data (pd.DataFrame): Bin data.
        distance_matrix (np.ndarray): Distance matrix.
        values (Dict): Parameter dictionary.

    Returns:
        float: Physical profit.
    """
    distance_route_vector = compute_distance_per_route(routes_list, distance_matrix)
    total_revenue = compute_waste_collection_revenue(routes_list, data, values["E"], values["B"], values["R"])
    transportation_cost = compute_transportation_cost(routes_list, distance_route_vector, values["C"])
    vehicle_penalty = compute_vehicle_use_penalty(routes_list, p_vehicle)

    profit = total_revenue - transportation_cost - vehicle_penalty
    return profit


def compute_total_profit(routes, distance_matrix, id_to_index, data, R, V, density, cost_per_km=1.0):
    """
    Calculate the total profit considering:
    - Revenue: sum of weights (kg) collected multiplied by R (€/kg)
    - Cost: sum of distances multiplied by cost per km (default = 1.0 €/km)

    Args:
        routes (List[List[int]]): Current routing solution.
        distance_matrix (np.ndarray): Distance matrix.
        id_to_index (Dict): Mapping of bin IDs to their indices.
        data (pd.DataFrame): Bin data.
        R (float): Revenue per kg.
        V (float): Bin volume.
        density (float): Bin density.
        cost_per_km (float, optional): Cost per km. Defaults to 1.0.

    Returns:
        float: Total calculated profit.
    """
    total_kg = 0
    total_km = 0
    stocks = dict(zip(data["#bin"], data["Stock"], strict=False))
    for route in routes:
        if len(route) <= 2:
            continue
        # Calculate profit
        total_kg += (sum(stocks.get(b, 0) for b in route if b != 0) / MAX_CAPACITY_PERCENT) * V * (density)
        # Use compute_sans_route_cost for distance calculation to reuse logic
        total_km += compute_sans_route_cost(route, distance_matrix, id_to_index)

    receita = total_kg * R
    custo = total_km * cost_per_km
    lucro = receita - custo
    return lucro, total_kg, total_km, receita
