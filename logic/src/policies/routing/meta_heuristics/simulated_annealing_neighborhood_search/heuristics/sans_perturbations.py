"""
SANS-specific mutation and perturbation operators.

Implements specialized operators for the SANS policy variant, including
destroy and repair heuristics (add/remove bins) and multi-node relocations
(n-move, n-swap) in both random and consecutive modes.

Attributes:
    remove_bins_from_route: Remove random bins from a single route.
    remove_n_bins_random: Remove n random bins from routes and add them to the removed pool.
    add_n_bins_random: Add n random bins from the removed pool back into routes.
    add_route_with_removed_bins_random: Create a new route from random bins in the removed pool.
    remove_n_bins_consecutive: Remove a sequence of consecutive bins from routes and add them to the removed pool.
    add_n_bins_consecutive: Add a sequence of consecutive bins from the removed pool back into routes.

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
from random import Random

from .sans_neighborhoods import insert_bin_in_route

__all__ = [
    "remove_bins_from_route",
    "remove_n_bins_random",
    "add_n_bins_random",
    "add_route_with_removed_bins_random",
    "remove_n_bins_consecutive",
    "add_n_bins_consecutive",
]


def remove_bins_from_route(routes_list: list, bins_cannot_removed: set, rng: Random, num_bins: int = 1) -> list:
    """
    Remove random bins from a single route.

    Args:
        routes_list: Solution routes list to mutate.
        bins_cannot_removed: Bins that cannot be removed.
        rng: Random number generator.
        num_bins: Number of bins to remove.

    Returns:
        New routes list with bins removed.
    """
    new_routes = copy.deepcopy(routes_list)
    valid_routes = [i for i, r in enumerate(new_routes) if len(r) > 2]

    if not valid_routes:
        return new_routes

    r_idx = rng.choice(valid_routes)
    route = new_routes[r_idx]

    valid_indices = [i for i in range(1, len(route) - 1) if route[i] not in bins_cannot_removed]
    num_to_remove = min(num_bins, len(valid_indices))

    if num_to_remove <= 0:
        return new_routes

    indices_to_remove = set(rng.sample(valid_indices, num_to_remove))
    new_routes[r_idx] = [n for i, n in enumerate(route) if i not in indices_to_remove]
    return new_routes


def remove_n_bins_random(
    routes_list: list, removed_bins: set, bins_cannot_removed: set, rng: Random, n: int = 2
) -> list:
    """
    Remove n random bins from routes and add them to the removed pool.

    Args:
        routes_list: Current routes.
        removed_bins: Set of removed bins to update.
        bins_cannot_removed: Bins that must stay in routes.
        rng: Random number generator.
        n: Number of bins to remove.

    Returns:
        Mutated solution.
    """
    new_routes = copy.deepcopy(routes_list)
    all_removable = []
    for r in new_routes:
        for b in r[1:-1]:
            if b not in bins_cannot_removed:
                all_removable.append(b)

    if not all_removable:
        return new_routes

    to_remove = rng.sample(all_removable, min(n, len(all_removable)))
    for b in to_remove:
        removed_bins.add(b)
        for _i, r in enumerate(new_routes):
            if b in r:
                r.remove(b)
                break
    return new_routes


def add_n_bins_random(
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


def add_route_with_removed_bins_random(
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


def remove_n_bins_consecutive(
    routes_list: list, removed_bins: set, bins_cannot_removed: set, rng: Random, n: int = 2
) -> list:
    """
    Remove a consecutive sequence of bins from a route.

    Args:
        routes_list: Current routes.
        removed_bins: Set of removed bins to update.
        bins_cannot_removed: Bins that must stay in routes.
        rng: Random number generator.
        n: Number of bins to remove.

    Returns:
        Mutated solution.
    """
    new_routes = copy.deepcopy(routes_list)
    valid_routes = [i for i, r in enumerate(new_routes) if len(r) >= n + 2]
    if not valid_routes:
        return new_routes

    r_idx = rng.choice(valid_routes)
    route = new_routes[r_idx]

    # Try to find a segment without 'mandatory' bins
    start_indices = list(range(1, len(route) - n))
    rng.shuffle(start_indices)

    for start in start_indices:
        segment = route[start : start + n]
        if not any(b in bins_cannot_removed for b in segment):
            for b in segment:
                removed_bins.add(b)
            new_routes[r_idx] = route[:start] + route[start + n :]
            break

    return new_routes


def add_n_bins_consecutive(
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
