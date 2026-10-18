"""
Intra-route node relocation operators.

Provides procedures to move single or multiple bins to different positions
within the same route, supporting both random and consecutive relocations.
Routes are depot-terminated (``[0, ..., 0]``).

Attributes:
    move_n_route_random: Moves n random bins to new positions within their route (n=1 moves one bin).
    move_n_route_consecutive: Moves a sequence of consecutive bins to a new position in the same route.
    relocate_within_route: Returns a copy of a route with one random bin moved to another position.
    or_opt_move: Returns a copy of a route with a random segment of 1-2 bins reinserted elsewhere.
    rearrange_part_route: Reorders a random segment of one route by nearest neighbour.

Example:
    >>> from logic.src.policies.helpers.operators.perturbation_shaking.sans.intra_move import move_n_route_random
    >>> routes = [[0, 1, 2, 3, 0], [0, 4, 5, 0]]
    >>> move_n_route_random(routes, rng, n=1)
    >>> print(routes)
"""

from random import Random
from typing import List, Optional

import numpy as np

from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import organize_route

__all__ = [
    "move_n_route_random",
    "move_n_route_consecutive",
    "relocate_within_route",
    "or_opt_move",
    "rearrange_part_route",
]


def move_n_route_random(routes_list: list, rng: Random, n: Optional[int] = None) -> Optional[int]:
    """
    Move n random bins to new positions within their routes.

    Args:
        routes_list: Current routing solution.
        rng: Random number generator.
        n: Number of nodes to move.

    Returns:
        The number of nodes moved.
    """
    if not routes_list:
        return None

    chosen_route = rng.sample(routes_list, 1)[0]
    if n is None:
        n = rng.sample([2, 3, 4, 5], 1)[0]

    if len(chosen_route) <= n + 2:
        return n

    for _ in range(n):
        # We need to re-sample every time because indices change
        # and we want to move 'n' nodes one after another as in the original code.
        bin_to_move = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

        target_bin = bin_to_move
        while target_bin == bin_to_move:
            target_bin = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

        target_idx = chosen_route.index(target_bin)
        chosen_route.remove(bin_to_move)
        chosen_route.insert(target_idx, bin_to_move)

    return n


def move_n_route_consecutive(routes_list: list, rng: Random, n: Optional[int] = None) -> Optional[int]:
    """
    Move a sequence of consecutive bins to a new position in the same route.

    Args:
        routes_list: Current routing solution.
        rng: Random number generator.
        n: Number of consecutive nodes to move.

    Returns:
        The number of nodes moved.
    """
    if not routes_list:
        return None

    chosen_route = rng.sample(routes_list, 1)[0]
    if n is None:
        n = rng.sample([2, 3, 4, 5], 1)[0]

    if len(chosen_route) <= n + 2:
        return n

    # Find a segment of size n (starting index should allow n nodes + depot)
    start_idx = rng.randint(1, len(chosen_route) - n - 1)
    segment = [chosen_route.pop(start_idx) for _ in range(n)]

    # Target node to insert AT
    target_bin = rng.sample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
    target_idx = chosen_route.index(target_bin)

    # Insert back at the new position
    for i, node in enumerate(segment):
        chosen_route.insert(target_idx + i, node)

    return n


def relocate_within_route(route: list, rng: Random) -> list:
    """
    Relocate a random bin to a different position in the same route.

    Args:
        route: Node sequence.
        rng: Random number generator.

    Returns:
        Mutated route.
    """
    if len(route) <= 3:
        return route[:]
    new_route = route[:]
    i, j = sorted(rng.sample(range(1, len(route) - 1), 2))
    bin_moved = new_route.pop(i)
    new_route.insert(j, bin_moved)
    return new_route


def or_opt_move(route: list, rng: Random) -> list:
    """
    Apply Or-opt operator (move a segment of 1-2 nodes) within a route.

    Args:
        route: Node sequence.
        rng: Random number generator.

    Returns:
        Mutated route.
    """
    if len(route) <= 4:
        return route[:]

    new_route = route[:]
    k = rng.choice([1, 2])
    start = rng.randint(1, len(route) - k - 1)
    segment = new_route[start : start + k]
    del new_route[start : start + k]

    insert_pos = rng.randint(1, len(new_route) - 1)
    new_route = new_route[:insert_pos] + segment + new_route[insert_pos:]
    return new_route


def rearrange_part_route(routes_list: List[List[int]], distance_matrix: np.ndarray, rng: Random) -> List[List[int]]:
    """
    Select a random portion of a route and reorder it using a greedy Nearest Neighbor heuristic.

    Args:
        routes_list: Current routing solution.
        distance_matrix: Distance matrix.
        rng: Random number generator.

    Returns:
        List[List[int]]: Routing solution with partially reordered route.
    """
    if not routes_list:
        return [[0]]

    chosen_route: List[int] = rng.sample(routes_list, 1)[0]
    length_chosen_route = len(chosen_route)

    if length_chosen_route < 4:
        return [[0]]

    possible_percent = [0.1, 0.2, 0.3, 0.4]
    chosen_n = rng.sample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_chosen_route)

    if chosen_n_percent < 2:
        return [[0]]

    bins = []

    max_start_index = length_chosen_route - 1 - chosen_n_percent
    if max_start_index < 1:
        return [[0]]

    start_index = rng.randint(1, max_start_index)
    segment = chosen_route[start_index : start_index + chosen_n_percent]

    # Remove segment from route
    bins = segment
    del chosen_route[start_index : start_index + chosen_n_percent]

    # Reorder bins using NN
    sorted_route_full = organize_route(bins, distance_matrix)
    # Remove depots (starts 0, ends 0)
    sorted_segment = sorted_route_full[1:-1]

    # Insert back at the same position
    for i, bin_id in enumerate(sorted_segment):
        chosen_route.insert(start_index + i, bin_id)

    routes_list[routes_list.index(chosen_route)] = chosen_route
    return routes_list
