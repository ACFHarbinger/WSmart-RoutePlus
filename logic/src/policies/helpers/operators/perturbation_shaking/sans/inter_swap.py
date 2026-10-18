"""
Inter-route swap operators.

Routes are depot-terminated (``[0, ..., 0]``).

Attributes:
    swap_n_2_routes_random: Swap n random bins, one pair at a time, between two routes (n=1 swaps one pair).
    swap_n_2_routes_consecutive: Swap two consecutive sequences of n bins between two routes.
    exchange_n_bins_random: Copy of the solution with n random bins exchanged between two routes.
    exchange_route_tails: Copy of the solution with the tails of two routes exchanged.

Example:
    >>> from logic.src.policies.helpers.operators.perturbation_shaking.sans.inter_swap import swap_n_2_routes_random
    >>> routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> swap_n_2_routes_random(routes, rng, n=1)
    >>> print(routes)
"""

import copy
from random import Random
from typing import List, Optional


def swap_n_2_routes_random(routes_list: List[List[int]], rng: Random, n: Optional[int] = None) -> Optional[int]:
    """
    Inter-route perturbation: Swap n random bins between two different routes.

    Args:
        routes_list: Current routing solution.
        rng: Random number generator.
        n: Number of bins to swap.

    Returns:
        int: The number of bins swapped (chosen_n).
    """
    chosen_n = n if n is not None else rng.sample([2, 3, 4, 5], 1)[0]

    if len(routes_list) > 1:
        chosen_routes = rng.sample(routes_list, 2)
        route1, route2 = chosen_routes

        if len(route1) >= chosen_n + 2 and len(route2) >= chosen_n + 2:
            for _ in range(chosen_n):
                bin1 = rng.sample(route1[1 : len(route1) - 1], 1)[0]
                bin2 = rng.sample(route2[1 : len(route2) - 1], 1)[0]

                pos1 = route1.index(bin1)
                pos2 = route2.index(bin2)

                route1[pos1], route2[pos2] = route2[pos2], route1[pos1]

    return chosen_n


def swap_n_2_routes_consecutive(
    routes_list: List[List[int]],
    rng: Random,
    n: Optional[int] = None,
    segment_may_cover_end_depot: bool = True,
) -> Optional[int]:
    """
    Inter-route perturbation: Swap two consecutive sequences of n bins
    between two different routes.

    Args:
        routes_list: Current routing solution (depot-terminated routes), mutated in place.
        rng: Random number generator.
        n: Number of consecutive nodes to swap.
        segment_may_cover_end_depot: If True (the original LAC operator), a segment may start at
            the last bin's successor, so it can carry a route's closing depot into the other route.
            If False, both segments lie strictly between the depots.

    Returns:
        int: The number of consecutive nodes swapped (chosen_n).
    """
    chosen_n = n if n is not None else rng.sample([2, 3, 4, 5], 1)[0]
    if len(routes_list) > 1:
        chosen_routes = rng.sample(routes_list, 2)
        route1, route2 = chosen_routes
        if len(route1) >= chosen_n + 2 and len(route2) >= chosen_n + 2:
            # Pick start indices for both segments
            last = 0 if segment_may_cover_end_depot else 1
            start1 = rng.randint(1, len(route1) - chosen_n - last)
            start2 = rng.randint(1, len(route2) - chosen_n - last)

            # Swap segments
            seg1 = route1[start1 : start1 + chosen_n]
            seg2 = route2[start2 : start2 + chosen_n]

            route1[start1 : start1 + chosen_n] = seg2
            route2[start2 : start2 + chosen_n] = seg1

    return chosen_n


def exchange_n_bins_random(routes_list: list, rng: Random, n: int = 2) -> list:
    """
    Exchange n random bins between two distinct routes, appending each route's incoming bins at its end.

    Args:
        routes_list: Current routing solution.
        rng: Random number generator.
        n: Number of nodes to swap.

    Returns:
        Mutated routing solution.
    """
    if len(routes_list) < 2:
        return copy.deepcopy(routes_list)

    new_routes = copy.deepcopy(routes_list)
    r1_idx, r2_idx = rng.sample(range(len(new_routes)), 2)
    route1, route2 = new_routes[r1_idx], new_routes[r2_idx]

    if len(route1) < n + 2 or len(route2) < n + 2:
        return new_routes

    nodes1 = rng.sample(route1[1:-1], n)
    nodes2 = rng.sample(route2[1:-1], n)

    # Perform swap
    new_routes[r1_idx] = [x for x in route1 if x not in nodes1] + nodes2
    new_routes[r2_idx] = [x for x in route2 if x not in nodes2] + nodes1

    # Keep depots (this implementation is a bit rough, let's fix it by maintaining depots)
    new_routes[r1_idx] = [0] + [x for x in new_routes[r1_idx] if x != 0] + [0]
    new_routes[r2_idx] = [0] + [x for x in new_routes[r2_idx] if x != 0] + [0]

    return new_routes


def exchange_route_tails(routes: list, rng: Random) -> list:
    """
    Exchange the tails of two different routes after a random cut in each.

    Args:
        routes: Set of routes.
        rng: Random number generator.

    Returns:
        Mutated solution.
    """
    if len(routes) < 2:
        return [r[:] for r in routes]

    new_routes = copy.deepcopy(routes)
    r1, r2 = rng.sample(range(len(new_routes)), 2)
    route1, route2 = new_routes[r1], new_routes[r2]
    if len(route1) <= 3 or len(route2) <= 3:
        return new_routes

    i1 = rng.randint(1, len(route1) - 2)
    i2 = rng.randint(1, len(route2) - 2)

    new_route1 = route1[:i1] + route2[i2:-1] + [0]
    new_route2 = route2[:i2] + route1[i1:-1] + [0]

    new_routes[r1] = new_route1
    new_routes[r2] = new_route2
    return new_routes
