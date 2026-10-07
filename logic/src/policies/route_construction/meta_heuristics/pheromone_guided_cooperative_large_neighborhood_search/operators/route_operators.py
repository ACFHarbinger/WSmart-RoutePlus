"""
Route-based local search operators for HGS.

(Refactored to point to `logic.src.policies.operators.route` package)

Attributes:
    move_swap_star: Swap two nodes in the same route.
    move_2opt_star: 2-opt move between two routes.
    move_2opt_intra: 2-opt move within a route.
    move_3opt_intra: 3-opt move within a route.

Example:
    >>> from logic.src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.route_operators import move_2opt_intra, move_2opt_star, move_3opt_intra, move_swap_star

    >>> routes = [[1, 2], [3, 4]]
    >>> move_2opt_intra(routes, 0, 0, 1)
    [[1, 2], [3, 4]]
    >>> move_2opt_star(routes, 0, 1)
    [[1, 3], [2, 4]]
    >>> move_3opt_intra(routes, 0, 0, 1, 1)
    [[1, 2], [3, 4]]
    >>> move_swap_star(routes, 0, 1)
    [[1, 3], [2, 4]]
"""

from .route import (
    move_2opt_intra,
    move_2opt_star,
    move_3opt_intra,
    move_swap_star,
)

__all__ = [
    "move_swap_star",
    "move_2opt_star",
    "move_2opt_intra",
    "move_3opt_intra",
]
