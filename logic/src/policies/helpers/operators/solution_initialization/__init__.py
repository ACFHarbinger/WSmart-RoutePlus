"""
Initialization schemes for VRPP.

Provides various constructive heuristics to build initial solutions:
- Greedy: Profit-driven insertion.
- Nearest Neighbor: Geographically compact clustering.
- Savings: Clarke-Wright savings algorithm.
- Regret: Regret-k insertion heuristic.
- GRASP: Randomized greedy adaptive search.
- Zones: LAC/SANS longitude-zone construction on depot-terminated routes.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.solution_initialization import build_greedy_routes
    >>> routes = build_greedy_routes(dist_matrix, wastes, capacity)
"""

from .grasp_si import build_grasp_routes
from .greedy_si import build_greedy_routes
from .nearest_neighbor_si import build_nn_routes, organize_route
from .regret_si import build_regret_routes
from .savings_si import build_savings_routes
from .zone_si import compute_initial_solution, find_initial_solution

__all__ = [
    "build_greedy_routes",
    "build_nn_routes",
    "build_savings_routes",
    "build_regret_routes",
    "build_grasp_routes",
    "organize_route",
    "find_initial_solution",
    "compute_initial_solution",
]
