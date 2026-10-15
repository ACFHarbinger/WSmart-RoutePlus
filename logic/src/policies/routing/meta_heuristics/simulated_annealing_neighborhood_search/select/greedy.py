"""
Greedy selection strategies for bin management.

Attributes:
    insert_bins: Greedily insert bins from the removed set into the current routes if it improves profit.
    remove_bins_end: Greedily remove bins from the current routes if it increases total profit.

Example:
    from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.select.greedy import (
        insert_bins,
        remove_bins_end,
    )
    routes_list = []
    removed_bins = []
    bins_cannot_removed = []
    data = pd.DataFrame()
    distance_matrix = np.ndarray()
    values = {}
    p_vehicle = 1.0
    p_load = 1.0
    p_route_difference = 1.0
    p_shift = 1.0
    insert_bins(routes_list, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
    remove_bins_end(routes_list, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, bins_cannot_removed, distance_matrix, values)
"""

from copy import deepcopy

from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.objectives import (
    compute_profit,
    compute_real_profit,
)


def insert_bins(
    routes_list,
    removed_bins,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    distance_matrix,
    values,
):
    """
    Greedily insert bins from the removed set into the current routes if it improves profit.

    Continues until no more profitable insertions are found.

    Args:
        routes_list: Initial routes.
        removed_bins: Available bins.
        p_vehicle: Vehicle penalty.
        p_load: Load penalty.
        p_route_difference: Difference penalty.
        p_shift: Shift penalty.
        data: Weights context.
        distance_matrix: Distance data.
        values: Constants.

    Returns:
        List[List[int]]: Optimized routing solution.
    """
    inserted_bins = []
    previous_solution = deepcopy(routes_list)
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
    for _it, a in enumerate(removed_bins):
        for it1, i in enumerate(previous_solution):
            i = routes_list[it1]
            for it2, _j in enumerate(i[0 : len(i) - 1]):
                i = routes_list[it1]
                for b in routes_list:
                    if a in b:
                        b.remove(a)
                position = it2 + 1
                i.insert(position, a)
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
                    previous_solution = deepcopy(routes_list)
                    previous_profit = new_profit
                else:
                    routes_list = deepcopy(previous_solution)
    for c in removed_bins:
        for m in previous_solution:
            if c in m:
                inserted_bins.append(c)

    for ins in inserted_bins:
        removed_bins.remove(ins)

    previous_real_profit = compute_real_profit(previous_solution, p_vehicle, data, distance_matrix, values)
    return previous_solution, previous_profit, previous_real_profit, removed_bins


def remove_bins_end(
    routes_list,
    removed_bins,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    bins_cannot_removed,
    distance_matrix,
    values,
):
    """
    Greedily remove bins from the current routes if it increases total profit.

    Args:
        routes_list: Initial routes.
        removed_bins: Target set for dropped bins.
        p_vehicle: Vehicle penalty.
        p_load: Load penalty.
        p_route_difference: Difference penalty.
        p_shift: Shift penalty.
        data: Bin metrics.
        bins_cannot_removed: Mandatory bins.
        distance_matrix: Metric matrix.
        values: Constants.

    Returns:
        List[List[int]]: Optimized routing solution.
    """
    previous_solution = deepcopy(routes_list)
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
    for it, i in enumerate(previous_solution):
        i = routes_list[it]
        for j in i[1 : len(i) - 1]:
            i = routes_list[it]
            if j not in bins_cannot_removed:
                i.remove(j)
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
                previous_solution = deepcopy(routes_list)
                previous_profit = new_profit
                removed_bins.append(j)
            else:
                routes_list = deepcopy(previous_solution)
    previous_real_profit = compute_real_profit(previous_solution, p_vehicle, data, distance_matrix, values)
    return previous_solution, previous_profit, previous_real_profit, removed_bins
