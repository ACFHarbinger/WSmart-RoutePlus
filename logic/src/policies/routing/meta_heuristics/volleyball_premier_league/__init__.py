"""
Volleyball Premier League (VPL) algorithm.

Reference:
    Moghdani, R., & Salimifard, K. (2018). "Volleyball Premier League
    Algorithm." Applied Soft Computing, 64, 161-185.
    DOI: https://doi.org/10.1016/j.asoc.2017.11.043

Attributes:
    VPLSolver: The VPL solver class.
    VPLPolicy: The VPL policy class.
    VPLConfig: The VPL parameters class.

Example:
    >>> solver = VPLSolver(params)
    >>> routes = solver.solve(dist, capacity, demand, nodes)
"""

from logic.src.configs.policies.routing.meta_heuristics.vpl import VPLConfig

from .policy_vpl import VPLPolicy
from .solver import VPLSolver

__all__ = ["VPLConfig", "VPLSolver", "VPLPolicy"]
