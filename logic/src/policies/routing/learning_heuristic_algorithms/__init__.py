"""
Module documentation.

Attributes:
    policy_rl_alns: Reinforcement Learning Adaptive Large Neighborhood Search policy.
    policy_rl_hvpl: Reinforcement Learning Hybrid Volleyball Premier League policy.
    policy_rl_gd_hh: Reinforcement Learning Great Deluge Hyper Heuristic policy.

Examples:
    >>> from logic.src.policies.routing.learning_heuristic_algorithms import policy_rl_alns
    >>> policy = policy_rl_alns()
    >>> routes, metrics = policy.run_day(env)
    >>> print(f"Best routes: {routes}")
    >>> print(f"Metrics: {metrics}")
"""

from .reinforcement_learning_adaptive_large_neighborhood_search import policy_rl_alns as policy_rl_alns
from .reinforcement_learning_great_deluge_hyper_heuristic import policy_rl_gd_hh as policy_rl_gd_hh
from .reinforcement_learning_hybrid_volleyball_premier_league import policy_rl_hvpl as policy_rl_hvpl

__all__ = [
    "policy_rl_alns",
    "policy_rl_hvpl",
    "policy_rl_gd_hh",
]
