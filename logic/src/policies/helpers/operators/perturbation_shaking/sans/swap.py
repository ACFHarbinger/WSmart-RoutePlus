"""
Node exchange operators for solution perturbation and neighborhood search.

Provides intra-route and inter-route bin swapping logic. Includes routines
for swapping single points or entire sequences (segments) between routes,
supporting randomized and consecutive exchange strategies.

Attributes:
    None

Example:
    None
"""

from .inter_swap import (
    exchange_n_bins_random,
    exchange_route_tails,
    swap_n_2_routes_consecutive,
    swap_n_2_routes_random,
)
from .intra_swap import (
    swap_n_route_consecutive,
    swap_n_route_random,
)

__all__ = [
    "swap_n_route_random",
    "swap_n_route_consecutive",
    "swap_n_2_routes_random",
    "swap_n_2_routes_consecutive",
    "exchange_n_bins_random",
    "exchange_route_tails",
]
