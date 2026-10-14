"""
Auxiliary modules for the Lin-Kernighan-Helsgaun 3 (LKH-3) policy.

Attributes:
    solve_lkh3: Main LKH-3 solver function for VRPP.

Example:
    >>> from logic.src.policies.routing.matheuristics.lin_kernighan_helsgaun_three import solve_lkh3
    >>> routes, profit, cost = solve_lkh3(dist_matrix, wastes, capacity, R, C, params)
"""

from .lkh3 import solve_lkh3

__all__ = ["solve_lkh3"]
