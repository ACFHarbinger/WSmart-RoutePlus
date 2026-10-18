"""
Random perturbations of depot-terminated routes (``[0, ..., 0]``), as used by SANS and LAC.

None of these operators checks for improvement: the calling annealer decides whether to accept.
The in-place operators mutate ``routes_list``; those documented as returning a copy leave it unchanged.

Attributes:
    move_n_route_random: Move n random bins within one route (n=1 moves one bin).
    move_n_route_consecutive: Move n consecutive bins within one route.
    relocate_within_route: Copy of a route with one random bin moved.
    or_opt_move: Copy of a route with a random segment of 1-2 bins reinserted.
    rearrange_part_route: Reorder a random segment of one route by nearest neighbour.
    move_n_2_routes_random: Move n random bins from one route to another (n=1 moves one bin).
    move_n_2_routes_consecutive: Move n consecutive bins from one route to another.
    move_between_routes: Every capacity-feasible single-bin move between two routes.
    transfer_n_bins_random: Copy with n random bins of one route inserted as a block into any route.
    transfer_n_bins_consecutive: Copy with n consecutive bins of one route inserted into any route.
    swap_n_route_random: Swap n random pairs of bins within one route (n=1 swaps one pair).
    swap_n_route_consecutive: Swap two segments of n bins within one route.
    swap_n_2_routes_random: Swap n random bins between two routes (n=1 swaps one pair).
    swap_n_2_routes_consecutive: Swap two segments of n bins between two routes.
    exchange_n_bins_random: Copy with n random bins exchanged between two routes.
    exchange_route_tails: Copy with the tails of two routes exchanged.

Example:
    >>> from logic.src.policies.helpers.operators.perturbation_shaking.sans import move_n_2_routes_random
    >>> routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> move_n_2_routes_random(routes, rng, n=1)
    >>> print(routes)
"""

from .move import (
    move_between_routes,
    move_n_2_routes_consecutive,
    move_n_2_routes_random,
    move_n_route_consecutive,
    move_n_route_random,
    or_opt_move,
    rearrange_part_route,
    relocate_within_route,
    transfer_n_bins_consecutive,
    transfer_n_bins_random,
)
from .swap import (
    exchange_n_bins_random,
    exchange_route_tails,
    swap_n_2_routes_consecutive,
    swap_n_2_routes_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)

__all__ = [
    "move_n_route_random",
    "move_n_route_consecutive",
    "relocate_within_route",
    "or_opt_move",
    "rearrange_part_route",
    "move_n_2_routes_random",
    "move_n_2_routes_consecutive",
    "move_between_routes",
    "transfer_n_bins_random",
    "transfer_n_bins_consecutive",
    "swap_n_route_random",
    "swap_n_route_consecutive",
    "swap_n_2_routes_random",
    "swap_n_2_routes_consecutive",
    "exchange_n_bins_random",
    "exchange_route_tails",
]
