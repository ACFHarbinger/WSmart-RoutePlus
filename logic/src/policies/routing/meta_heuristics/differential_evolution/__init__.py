r"""Differential Evolution (DE) policy for VRPP.

This module implements the DE/rand/1/bin algorithm for solving the VRPP,
providing a robust population-based search strategy.

Attributes:
    DESolver: Core solver class for Differential Evolution.
    DEConfig: Parameter configuration for DE.
    DEPolicy: Policy adapter for DE (imported in policy_de.py).

Example:
    >>> from logic.src.policies.routing.meta_heuristics.differential_evolution import DESolver
    >>> solver = DESolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

from logic.src.configs.policies.routing.meta_heuristics.de import DEConfig

from .solver import DESolver

__all__ = ["DEConfig", "DESolver"]
