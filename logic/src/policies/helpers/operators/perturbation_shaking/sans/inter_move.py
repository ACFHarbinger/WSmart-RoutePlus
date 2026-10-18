"""
Inter-route node relocation operators.

Provides procedures to move single or multiple bins between different routes,
supporting both random and consecutive relocations. Routes are depot-terminated
(``[0, ..., 0]``).

Attributes:
    move_n_2_routes_random: Move n random bins, one at a time, from one route to another (n=1 moves one bin).
    move_n_2_routes_consecutive: Move n consecutive bins from one route to another.
    move_between_routes: List every capacity-feasible single-bin move between two routes.
    transfer_n_bins_random: Copy of the solution with n random bins of one route inserted as a block into any route.
    transfer_n_bins_consecutive: Copy of the solution with n consecutive bins of one route inserted into any route.
    add_route_random: Move a random share of one route's bins to a new nearest-neighbour route.
    add_route_consecutive: Move a consecutive share of one route's bins to a new nearest-neighbour route.

Example:
    >>> from logic.src.policies.helpers.operators.perturbation_shaking.sans.inter_move import move_n_2_routes_random
    >>> routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> move_n_2_routes_random(routes, rng, n=1)
    >>> print(routes)
"""

import copy
from random import Random
from typing import List, Optional

from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import organize_route

__all__ = [
    "move_n_2_routes_random",
    "move_n_2_routes_consecutive",
    "move_between_routes",
    "transfer_n_bins_random",
    "transfer_n_bins_consecutive",
    "add_route_random",
    "add_route_consecutive",
]


def move_n_2_routes_random(routes_list: List[List[int]], rng: Random, n: Optional[int] = None) -> Optional[int]:
    """
    Inter-route perturbation: Move n random bins from one route to another.

    Args:
        routes_list: Current routing solution.
        rng: Random number generator.
        n: Number of bins to move.

    Returns:
        int: The number of bins moved (chosen_n).
    """
    if len(routes_list) < 2:
        return None

    # Pick n
    n = n if n is not None else rng.sample([2, 3, 4, 5], 1)[0]

    # Select donor and receptor
    r1, r2 = rng.sample(routes_list, 2)
    donor_route, receptor_route = rng.sample([r1, r2], 2)

    if len(donor_route) < n + 2:
        return n

    for _ in range(n):
        # Sample donor bin
        donor_bin = rng.sample(donor_route[1 : len(donor_route) - 1], 1)[0]

        # Sample receptor position
        if len(receptor_route) > 2:
            receptor_position = receptor_route.index(rng.sample(receptor_route[1 : len(receptor_route) - 1], 1)[0])
        else:
            receptor_position = 1

        donor_route.remove(donor_bin)
        receptor_route.insert(receptor_position, donor_bin)

    # Clean up
    if len(donor_route) == 2:
        routes_list.remove(donor_route)

    return n


def move_n_2_routes_consecutive(routes_list: List[List[int]], rng: Random, n: Optional[int] = None) -> Optional[int]:
    """
    Inter-route perturbation: Move a sequence of consecutive bins from one route
    to another.

    Args:
        routes_list: Current routing solution.
        rng: Random number generator.
        n: Number of consecutive nodes to move.

    Returns:
        int: The number of nodes moved (chosen_n).
    """
    if len(routes_list) < 2:
        return None

    n = n if n is not None else rng.sample([2, 3, 4, 5], 1)[0]

    r1, r2 = rng.sample(routes_list, 2)
    donor_route, receptor_route = rng.sample([r1, r2], 2)

    if len(donor_route) < n + 2:
        return n

    # Find segment in donor
    start_idx = rng.randint(1, len(donor_route) - n - 1)
    segment = [donor_route.pop(start_idx) for _ in range(n)]

    # Target position in receptor
    if len(receptor_route) > 2:
        target_node = rng.sample(receptor_route[1 : len(receptor_route) - 1], 1)[0]
        target_idx = receptor_route.index(target_node)
    else:
        target_idx = 1

    # Insert back
    for i, node in enumerate(segment):
        receptor_route.insert(target_idx + i, node)

    # Clean up
    if len(donor_route) == 2:
        routes_list.remove(donor_route)

    return n


def move_between_routes(routes: list, data, vehicle_capacity: float, id_to_index: dict, rng: Random) -> list:
    """
    Move a random bin from one route to another, respecting capacity constraints.

    Args:
        routes: Set of routes.
        data: Bin weights data.
        vehicle_capacity: Tanker capacity.
        id_to_index: Mapping.
        rng: Random number generator.

    Returns:
        List of mutated solution candidates.
    """
    moves = []
    # Import locally to avoid circularity if any
    try:
        stocks = dict(zip(data["#bin"], data["Stock"], strict=False))
    except (KeyError, TypeError):
        # Fallback if data format is different
        return []

    for i in range(len(routes)):
        for j in range(len(routes)):
            if i == j:
                continue
            for idx in range(1, len(routes[i]) - 1):
                bin_to_move = routes[i][idx]
                if bin_to_move == 0:
                    continue
                load_j = sum(stocks.get(b, 0) for b in routes[j] if b != 0)
                if load_j + stocks.get(bin_to_move, 0) > vehicle_capacity:
                    continue
                new_routes = copy.deepcopy(routes)
                new_routes[i].pop(idx)
                insert_pos = rng.randint(1, len(new_routes[j]) - 1)
                new_routes[j].insert(insert_pos, bin_to_move)
                moves.append(new_routes)
    return moves


def transfer_n_bins_random(routes_list: list, rng: Random, n: int = 2) -> list:
    """
    Move n random bins of one route, as a block, to a random position of any route.

    The receiving route may be the donor itself.

    Args:
        routes_list: Current routing solution.
        rng: Random number generator.
        n: Number of nodes to move.

    Returns:
        Mutated routing solution.
    """
    new_routes = copy.deepcopy(routes_list)
    non_empty = [r for r in new_routes if len(r) > 2]
    if not non_empty:
        return new_routes

    donor_route = rng.choice(non_empty)
    if len(donor_route) < n + 2:
        return new_routes

    nodes_to_move = rng.sample(donor_route[1:-1], n)
    # Filter donor route
    donor_idx = new_routes.index(donor_route)
    new_routes[donor_idx] = [x for x in donor_route if x not in nodes_to_move]

    # Receptor route
    receptor_route = rng.choice([r for r in new_routes if len(r) >= 2])
    receptor_idx = new_routes.index(receptor_route)
    insert_pos = rng.randint(1, max(1, len(receptor_route) - 1))

    # Insert nodes (in order)
    new_routes[receptor_idx] = receptor_route[:insert_pos] + nodes_to_move + receptor_route[insert_pos:]
    return new_routes


def transfer_n_bins_consecutive(routes_list: list, rng: Random, n: int = 2) -> list:
    """
    Move a sequence of consecutive bins to a random position of any route.

    The receiving route may be the donor itself.

    Args:
        routes_list: Current routes.
        rng: Random number generator.
        n: Number of bins to move.

    Returns:
        Mutated solution.
    """
    new_routes = copy.deepcopy(routes_list)
    non_empty = [r for r in new_routes if len(r) >= n + 2]
    if not non_empty:
        return new_routes

    donor_route = rng.choice(non_empty)
    donor_idx = new_routes.index(donor_route)
    start_idx = rng.randint(1, len(donor_route) - n - 1)
    segment = donor_route[start_idx : start_idx + n]

    # Remove from donor
    new_donor = donor_route[:start_idx] + donor_route[start_idx + n :]
    new_routes[donor_idx] = new_donor

    # Receptor
    receptor_route = rng.choice(new_routes)
    receptor_idx = new_routes.index(receptor_route)
    insert_pos = rng.randint(1, max(1, len(receptor_route) - 1))
    new_routes[receptor_idx] = receptor_route[:insert_pos] + segment + receptor_route[insert_pos:]

    return new_routes


# Add one route random
def add_route_random(routes_list, distance_matrix, rng):
    """
    Add a new route with a single bin chosen randomly from existing routes.

    Args:
        routes_list: All routes.
        distance_matrix: For sequencing.
        rng: Random number generator.

    Returns:
        List[List[int]]: Updated set of routes.
    """
    if len(routes_list) > 0:
        chosen_route = rng.sample(routes_list, 1)[0]
    else:
        return routes_list

    length_chosen_route = len(chosen_route)
    possible_percent = [0.3, 0.4, 0.5, 0.6]
    chosen_n = rng.sample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_chosen_route)
    bins = []
    for _s in range(0, chosen_n_percent):
        chosen_bin = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
        chosen_route.remove(chosen_bin)
        bins.append(chosen_bin)

    route = organize_route(bins, distance_matrix)
    routes_list.append(route)
    if len(chosen_route) == 2:
        chosen_route = []

    routes_list = list(filter(None, routes_list))
    return chosen_n


# Add one route consecutive
def add_route_consecutive(routes_list, distance_matrix, rng):
    """
    Create a new route by extracting a consecutive segment from an existing route.

    Args:
        routes_list: Current routes.
        distance_matrix: Distance matrix for route organization.
        rng: Random number generator.

    Returns:
        int: The number of bins moved to the new route.
    """
    if len(routes_list) > 0:
        chosen_route = rng.sample(routes_list, 1)[0]
    else:
        return 0

    length_chosen_route = len(chosen_route)
    possible_percent = [0.3, 0.4, 0.5, 0.6]
    chosen_n = rng.sample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_chosen_route)
    bins = []

    position_chosen_bin = 0
    for s in range(0, chosen_n_percent):
        if s == 0:
            chosen_bin = rng.sample(chosen_route[1 : len(chosen_route) - chosen_n_percent], 1)[0]
            position_chosen_bin = chosen_route.index(chosen_bin)
        else:
            chosen_bin = chosen_route[position_chosen_bin + s]

        bins.append(chosen_bin)

    chosen_bins = bins.copy()
    route = organize_route(bins, distance_matrix)
    routes_list.append(route)
    for u in chosen_bins:
        chosen_route.remove(u)

    if len(chosen_route) == 2:
        chosen_route = []

    routes_list = list(filter(None, routes_list))
    return chosen_n
