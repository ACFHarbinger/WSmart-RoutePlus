r"""(μ,κ,λ) Evolution Strategy with age-based selection.

This module implements the classical (μ,κ,λ)-ES where selection occurs from
μ parents who have not exceeded an age of κ and λ offspring individuals.

Attributes:
    MuKappaLambdaESSolver: Main solver class.
    MuKappaLambdaESConfig: Configuration parameters.
    Individual: Data structure for ES individuals with age tracking.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.evolution_strategy_mu_kappa_lambda import MuKappaLambdaESSolver
    >>> solver = MuKappaLambdaESSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()

Reference:
    Emmerich, M., Shir, O. M., & Wang, H. (2015). Evolution Strategies.
    In: Handbook of Natural Computing (pages 1-31).
"""

from logic.src.configs.policies.es_mkl import MuKappaLambdaESConfig
from .solver import Individual, MuKappaLambdaESSolver

__all__ = [
    "MuKappaLambdaESSolver",
    "MuKappaLambdaESConfig",
    "Individual",
]
