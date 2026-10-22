"""
Variable Neighborhood Search (VNS) solver for VRPP.

Attributes:
    VNSSolver: The main solver class.
    VNSConfig: Configuration parameters dataclass.

Example:
    >>> solver = VNSSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

from logic.src.configs.policies.routing.meta_heuristics.vns import VNSConfig

from .solver import VNSSolver

__all__ = ["VNSSolver", "VNSConfig"]
