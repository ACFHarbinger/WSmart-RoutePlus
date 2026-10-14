"""
Variable Neighborhood Search (VNS) solver for VRPP.

Attributes:
    VNSSolver: The main solver class.
    VNSParams: Configuration parameters dataclass.

Example:
    >>> solver = VNSSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

from .params import VNSParams
from .solver import VNSSolver

__all__ = ["VNSSolver", "VNSParams"]
