"""
Operators for local search moves and swaps.

Attributes:
    move_1_route: Move one bin from one route to another.
    move_2_routes: Move one bin from one route to another.
    move_n_2_routes_consecutive: Move n consecutive bins from one route to another.
    move_n_2_routes_random: Move n random bins from one route to another.
    move_n_route_consecutive: Move n consecutive bins from one route to another.
    move_n_route_random: Move n random bins from one route to another.
    swap_1_route: Swap one bin from one route with another bin from a different route.
    swap_2_routes: Swap one bin from one route with another bin from a different route.
    swap_n_2_routes_consecutive: Swap two consecutive sequences of n bins between two different routes.
    swap_n_2_routes_random: Swap n random bins between two different routes.
    swap_n_route_consecutive: Swap two consecutive sequences of n bins between two different routes.
    swap_n_route_random: Swap n random bins between two different routes.

Example:
    >>> from logic.src.policies.helpers.operators.perturbation_shaking.sans import move_1_route, move_2_routes, move_n_2_routes_consecutive, move_n_2_routes_random, move_n_route_consecutive, move_n_route_random, swap_1_route, swap_2_routes, swap_n_2_routes_consecutive, swap_n_2_routes_random, swap_n_route_consecutive, swap_n_route_random
    >>> routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> move_1_route(routes, rng)
    >>> print(routes)
"""

from .move import (
    move_1_route,
    move_2_routes,
    move_n_2_routes_consecutive,
    move_n_2_routes_random,
    move_n_route_consecutive,
    move_n_route_random,
)
from .swap import (
    swap_1_route,
    swap_2_routes,
    swap_n_2_routes_consecutive,
    swap_n_2_routes_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)

__all__ = [
    "move_1_route",
    "move_2_routes",
    "move_n_2_routes_consecutive",
    "move_n_2_routes_random",
    "move_n_route_consecutive",
    "move_n_route_random",
    "swap_1_route",
    "swap_2_routes",
    "swap_n_2_routes_consecutive",
    "swap_n_2_routes_random",
    "swap_n_route_consecutive",
    "swap_n_route_random",
]
