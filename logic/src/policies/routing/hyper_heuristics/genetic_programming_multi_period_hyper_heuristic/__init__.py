"""
GP-MP-HH (Genetic Programming Multi-Period Hyper-Heuristic) policy module.

Reference:
    Bhandari, R., & Keedwell, E. (2021). "A genetic programming multi-period
    selection hyper-heuristic for the vehicle routing problem with time windows."
    In Proceedings of the 2021 Annual Conference on Genetic and Evolutionary
    Computation (pp. 355-363).

This module implements the GP-MP-HH policy, which is a multi-period selection
selection hyper-heuristic that uses genetic programming to evolve a sequence
of low-level heuristics that are applied sequentially to an initial solution.

Attributes:
    GPMPHHPolicy: Implements the GP-MP-HH policy.
    GP_MP_HH_Params: Parameters for the GP-MP-HH policy.

Example:
    >>> solver = GPMPHHPolicy(config)
    >>> solution = solver.solve()
    >>> print(solution)
    SolutionContext(...)
"""

from .params import GP_MP_HH_Params
from .policy_gp_mp_hh import GPMPHHPolicy

__all__ = ["GPMPHHPolicy", "GP_MP_HH_Params"]
