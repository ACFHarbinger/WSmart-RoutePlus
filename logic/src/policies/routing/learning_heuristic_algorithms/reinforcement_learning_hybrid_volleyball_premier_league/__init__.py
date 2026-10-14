"""
Reinforcement Learning Hybrid Volleyball Premier League (RL-HVPL).

A middle-ground metaheuristic combining:
    - Enhanced ACO with Q-Learning for construction
    - Enhanced ALNS with SARSA for improvement
    - VPL population framework for diversity

This bridges HVPL (basic) and RL-AHVPL (advanced with HGS/CMAB/GLS).

Attributes:
    RLHVPLParams (RLHVPLParams): RL-HVPL parameters class.
    RLHVPLSolver (RLHVPLSolver): RL-HVPL solver class.

Example:
    >>> from logic.src.policies.routing.learning_heuristic_algorithms
    ...     .reinforcement_learning_hybrid_volleyball_premier_league import (
    ...         RLHVPLParams, RLHVPLSolver
    ...     )
    >>> solver = RLHVPLSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
    >>> print(routes)
    >>> print(profit)
    >>> print(cost)
"""

from .params import RLHVPLParams
from .rl_hvpl import RLHVPLSolver

__all__ = ["RLHVPLParams", "RLHVPLSolver"]
