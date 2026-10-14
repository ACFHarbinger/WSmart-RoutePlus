"""
RL-ALNS: Reinforcement Learning-augmented Adaptive Large Neighborhood Search.

This package implements online RL algorithms for dynamic operator selection in ALNS.

Based on research: "Online Reinforcement Learning for Inference-Time Operator Selection
in the Stochastic Multi-Period Capacitated Vehicle Routing Problem"

Attributes:
-----------
    RLALNSSolver: The main solver class.
    RLALNSParams: The configuration class for the solver.

Example:
--------
    >>> from logic.src.policies.routing.learning_heuristic_algorithms import RLALNSSolver
    >>> from logic.src.policies.routing.learning_heuristic_algorithms import RLALNSParams
    >>> policy = RLALNSSolver(RLALNSParams())
    >>> routes, metrics = policy.run_day(env)
    >>> print(f"Best routes: {routes}")
    >>> print(f"Metrics: {metrics}")
"""

from .params import RLALNSParams
from .solver import RLALNSSolver

__all__ = ["RLALNSSolver", "RLALNSParams"]
