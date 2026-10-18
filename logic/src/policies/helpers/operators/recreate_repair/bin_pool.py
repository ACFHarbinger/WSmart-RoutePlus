"""
Random reinsertion of bins from a pool of removed bins into depot-terminated routes.

These are the insertion moves of the LAC and SANS annealers: random perturbations
without an improvement check. The LAC operators mutate ``routes_list`` and the
``removed_bins`` list without checking capacity; the SANS operators return a
copy of the routes, check capacity, and discard inserted bins from a set.

Attributes:
    add_n_bins_random: (LAC) Add n random removed bins at random positions (n=1 adds one bin).
    add_n_bins_consecutive: (LAC) Add n random removed bins as a block at a random position.
    add_route_with_removed_bins_random: (LAC) New nearest-neighbour route from a random share of the pool.
    add_route_with_removed_bins_consecutive: (LAC) New nearest-neighbour route from a contiguous share of the pool.
    insert_bin_in_route: (SANS) Copy of a route with one bin at its cheapest position.
    insert_n_bins_cheapest: (SANS) Copy with n removed bins each placed cheapest in a random route if it fits.
    insert_n_bins_block: (SANS) Copy with n removed bins placed as a block in a random route if they fit.
    add_route_from_pool_capacitated: (SANS) Copy with a new route filled from the shuffled pool up to capacity.

Example:
    >>> from logic.src.policies.helpers.operators.recreate_repair.bin_pool import add_n_bins_random
    >>> routes, removed = [[0, 1, 2, 0]], [3]
    >>> add_n_bins_random(routes, removed, rng, n=1)
"""

import copy
from random import Random

from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import organize_route


def add_n_bins_random(routes_list, removed_bins, rng, n=None):
    """
    Add n random bins from the removed list, each at a random position of a random route.

    Args:
        routes_list: Current routes (depot-terminated), mutated in place.
        removed_bins: Removed bins, reduced in place.
        rng: Random number generator.
        n: Number of bins to add; drawn from 2-5 when None.

    Returns:
        Tuple[List[int], int]: List of added bins and n.
    """
    bins_to_add_random = []
    possible_n = [2, 3, 4, 5]
    chosen_n = n if n is not None else rng.sample(possible_n, 1)[0]
    if len(routes_list) > 0 and len(removed_bins) >= chosen_n:
        for _b in range(0, chosen_n):
            bin_to_add = rng.sample(removed_bins, 1)[0]
            chosen_route = rng.sample(routes_list, 1)[0]
            chosen_position = chosen_route.index(rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0])
            chosen_route.insert(chosen_position, bin_to_add)
            removed_bins.remove(bin_to_add)
            bins_to_add_random.append(bin_to_add)
    return bins_to_add_random, chosen_n


# Function to add n bin from the removed bins set to any route randomly
def add_n_bins_consecutive(routes_list, removed_bins, rng):
    """
    Add n consecutive bins from the removed set (treated as list) to a random route.

    Args:
        routes_list: Current routes.
        removed_bins: Set/List of removed bins.
        rng: Random number generator.

    Returns:
        Tuple[List[int], int]: List of added bins and the number n chosen.
    """
    bins_to_add_consecutive = []
    possible_n = [2, 3, 4, 5]
    chosen_n = rng.sample(possible_n, 1)[0]
    if len(routes_list) > 0 and len(removed_bins) >= chosen_n:
        for _b in range(0, chosen_n):
            bin_to_add = rng.sample(removed_bins, 1)[0]
            removed_bins.remove(bin_to_add)
            bins_to_add_consecutive.append(bin_to_add)

        chosen_route = rng.sample(routes_list, 1)[0]
        chosen_position = chosen_route.index(rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0])
        for d in range(0, chosen_n):
            chosen_route.insert(chosen_position + d, bins_to_add_consecutive[d])
    return bins_to_add_consecutive, chosen_n


# Add one route with bins from removed bins
def add_route_with_removed_bins_random(routes_list, removed_bins, distance_matrix, rng):
    """
    Create a new route using a random subset of removed bins.

    Args:
        routes_list: Current routes.
        removed_bins: Pool of removed bins.
        distance_matrix: Distance matrix.
        rng: Random number generator.

    Returns:
        Tuple[int, List[int]]: Number of bins used and the list of bins used.
    """
    length_removed_bins = len(removed_bins)
    possible_percent = [0.2, 0.3, 0.4, 0.5]
    chosen_n = rng.sample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_removed_bins)
    bins_random = []
    for _o in range(0, chosen_n_percent):
        chosen_bin = rng.sample(removed_bins, 1)[0]
        removed_bins.remove(chosen_bin)
        bins_random.append(chosen_bin)

    chosen_bins = bins_random.copy()
    route = organize_route(chosen_bins, distance_matrix)
    routes_list.append(route)
    return chosen_n, bins_random


# Add one route with bins from removed bins consecutive
def add_route_with_removed_bins_consecutive(routes_list, removed_bins, distance_matrix, rng):
    """
    Create a new route using a consecutive sequence of removed bins.

    Args:
        routes_list: Current routes.
        removed_bins: Pool of removed bins.
        distance_matrix: Distance matrix.
        rng: Random number generator.

    Returns:
        Tuple[int, List[int]]: Number of bins used and the list of bins used.
    """
    length_removed_bins = len(removed_bins)
    possible_percent = [0.2, 0.3, 0.4, 0.5]
    chosen_n = rng.sample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_removed_bins)
    bins_consecutive = []

    position_chosen_bin = 0
    for o in range(0, chosen_n_percent):
        if o == 0:
            chosen_bin = rng.sample(removed_bins[0 : len(removed_bins) - chosen_n_percent + 1], 1)[0]
            position_chosen_bin = removed_bins.index(chosen_bin)
        else:
            chosen_bin = removed_bins[position_chosen_bin + o]

        bins_consecutive.append(chosen_bin)

    chosen_bins = bins_consecutive.copy()
    chosen_bins_organize = bins_consecutive.copy()
    route = organize_route(chosen_bins_organize, distance_matrix)
    routes_list.append(route)
    for u in chosen_bins:
        removed_bins.remove(u)
    return chosen_n, bins_consecutive


def insert_bin_in_route(route: list, bin_id: int, id_to_index: dict, distance_matrix) -> list:
    """
    Insert a bin into a route at the position minimizing cost increase.

    Args:
        route: Current route.
        bin_id: Bin to insert.
        id_to_index: Mapping from bin ID to matrix index.
        distance_matrix: Distance matrix.

    Returns:
        New route with the inserted bin.
    """
    best_route = route[:]
    best_increase = float("inf")

    # Start and end positions for insertion (between depots)
    for i in range(1, len(route)):
        prev_node = route[i - 1]
        next_node = route[i]

        idx_prev = id_to_index.get(prev_node, prev_node)
        idx_next = id_to_index.get(next_node, next_node)
        idx_bin = id_to_index.get(bin_id, bin_id)

        try:
            increase = (
                distance_matrix[idx_prev][idx_bin]
                + distance_matrix[idx_bin][idx_next]
                - distance_matrix[idx_prev][idx_next]
            )
        except (IndexError, TypeError):
            continue

        if increase < best_increase:
            best_increase = increase
            best_route = route[:i] + [bin_id] + route[i:]

    return best_route


def insert_n_bins_cheapest(
    routes_list: list,
    removed_bins: set,
    stocks: dict,
    vehicle_capacity: float,
    id_to_index: dict,
    distance_matrix,
    rng: Random,
    n: int = 2,
) -> list:
    """
    Add n random bins from the removed pool back into routes.

    Args:
        routes_list: Current routes.
        removed_bins: Set of removed bins to update.
        stocks: Dictionary of bin stocks.
        vehicle_capacity: Vehicle capacity.
        id_to_index: Mapping from bin IDs to their indices.
        distance_matrix: Distance matrix.
        rng: Random number generator.
        n: Number of bins to add.

    Returns:
        Mutated solution.
    """
    new_routes = copy.deepcopy(routes_list)
    if not new_routes or not removed_bins:
        return new_routes

    bins_to_add = rng.sample(sorted(list(removed_bins)), min(n, len(removed_bins)))
    for b in bins_to_add:
        # Try a random route
        r_idx = rng.randint(0, len(new_routes) - 1)
        route = new_routes[r_idx]
        load = sum(stocks.get(x, 0) for x in route if x != 0)
        if load + stocks.get(b, 0) <= vehicle_capacity:
            new_routes[r_idx] = insert_bin_in_route(route, b, id_to_index, distance_matrix)
            removed_bins.discard(b)
    return new_routes


def insert_n_bins_block(
    routes_list: list,
    removed_bins: set,
    stocks: dict,
    vehicle_capacity: float,
    id_to_index: dict,
    distance_matrix,
    rng: Random,
    n: int = 2,
) -> list:
    """
    Add a sequence of bins from removed set as a consecutive segment.

    Args:
        routes_list: Current routes.
        removed_bins: Set of removed bins to update.
        stocks: Dictionary of bin stocks.
        vehicle_capacity: Vehicle capacity.
        id_to_index: Mapping from bin IDs to their indices.
        distance_matrix: Distance matrix.
        rng: Random number generator.
        n: Number of bins to add.

    Returns:
        Mutated solution.
    """
    new_routes = copy.deepcopy(routes_list)
    if len(removed_bins) < n or not new_routes:
        return new_routes

    bins_to_add = rng.sample(sorted(list(removed_bins)), n)
    total_weight = sum(stocks.get(b, 0) for b in bins_to_add)

    r_idx = rng.randint(0, len(new_routes) - 1)
    route = new_routes[r_idx]
    load = sum(stocks.get(x, 0) for x in route if x != 0)

    if load + total_weight <= vehicle_capacity:
        # Insert them consecutively at a random position
        pos = rng.randint(1, len(route) - 1)
        new_routes[r_idx] = route[:pos] + bins_to_add + route[pos:]
        for b in bins_to_add:
            removed_bins.discard(b)

    return new_routes


def add_route_from_pool_capacitated(
    routes_list: list, removed_bins: set, stocks: dict, vehicle_capacity: float, rng: Random
) -> list:
    """
    Create a new route from random bins in the removed pool.

    Args:
        routes_list: Current routes.
        removed_bins: Set of removed bins to update.
        stocks: Dictionary of bin stocks.
        vehicle_capacity: Vehicle capacity.
        rng: Random number generator.

    Returns:
        Mutated solution.
    """
    if not removed_bins:
        return copy.deepcopy(routes_list)

    new_routes = copy.deepcopy(routes_list)
    new_route = [0]
    current_load = 0

    available = sorted(list(removed_bins))
    rng.shuffle(available)

    for b in available:
        b_weight = stocks.get(b, 0)
        if current_load + b_weight <= vehicle_capacity:
            new_route.append(b)
            current_load += b_weight
            removed_bins.discard(b)

    if len(new_route) > 1:
        new_route.append(0)
        new_routes.append(new_route)

    return new_routes


def insert_bins(routes_list, removed_bins, profit_fn, real_profit_fn):
    """
    Greedily insert bins from the removed list wherever the objective improves.

    Each removed bin is tried after every stop of every route, keeping each insertion that
    raises ``profit_fn``; bins that end up in a route leave ``removed_bins``.

    Args:
        routes_list: Initial routes (depot-terminated).
        removed_bins: Removed bins, reduced in place.
        profit_fn: Objective to maximise, such as the penalised profit.
        real_profit_fn: Objective reported for the result, such as the unpenalised profit.

    Returns:
        Tuple[List[List[int]], float, float, List[int]]: Solution, its ``profit_fn``, its
        ``real_profit_fn`` and the remaining removed bins.
    """
    inserted_bins = []
    previous_solution = copy.deepcopy(routes_list)
    previous_profit = profit_fn(previous_solution)
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
                new_profit = profit_fn(routes_list)
                if new_profit > previous_profit:
                    previous_solution = copy.deepcopy(routes_list)
                    previous_profit = new_profit
                else:
                    routes_list = copy.deepcopy(previous_solution)
    for c in removed_bins:
        for m in previous_solution:
            if c in m:
                inserted_bins.append(c)
    for ins in inserted_bins:
        removed_bins.remove(ins)
    previous_real_profit = real_profit_fn(previous_solution)
    return previous_solution, previous_profit, previous_real_profit, removed_bins
