"""
GIHH (Hyper-Heuristic with Two Guidance Indicators) policy module.

This module implements a selection hyper-heuristic that uses two guidance
indicators to adaptively select low-level heuristics during search:
1. Improvement Rate Indicator (IRI): Measures solution quality improvement
2. Time-based Indicator (TBI): Measures computational efficiency

Reference:
    Kheiri, A., & Keedwell, E. (2015). A sequence-based selection hyper-heuristic
    utilising a hidden Markov model. In Proceedings of the 2015 Annual Conference
    on Genetic and Evolutionary Computation (pp. 417-424).

Attributes:
    GIHHSolver: Solver for the GIHH policy.
    GIHHParams: Parameters for the GIHH policy.

Example:
    >>> solver = GIHHSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> solutions = solver.solve()
    >>> best_solution = max(solutions, key=lambda s: s.profit)
    >>> print(best_solution.profit)
    231.0
"""

from .gihh import GIHHSolver
from .params import GIHHParams

__all__ = ["GIHHSolver", "GIHHParams"]
