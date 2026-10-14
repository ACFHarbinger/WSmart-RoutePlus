"""
Search sub-package export for Simulated Annealing Neighborhood Search (SANS).

Attributes:
    local_search: Apply a randomized set of local search operators to improve a solution.
    local_search_2: Apply a deterministic set of local search operators to improve a solution.
    local_search_reversed: Apply a deterministic set of local search operators to improve a solution using reversed routes.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.search import local_search_2
    >>> routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
    >>> cost, profit = local_search_2(routes, distance_matrix, values, capacities, mandatory_nodes)
    >>> print(f"Cost: {cost}, Profit: {profit}")
"""

from .deterministic import local_search_2
from .random_search import local_search
from .reversed import local_search_reversed

__all__ = ["local_search", "local_search_2", "local_search_reversed"]
