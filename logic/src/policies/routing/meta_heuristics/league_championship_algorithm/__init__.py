"""
League Championship Algorithm (LCA) for VRPP.

LCA is a population-based metaheuristic inspired by the competition in a
sports league.

Attributes:
    LCAConfig: Configuration parameters for LCA.
    LCAPolicy: Policy adapter for LCA.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.league_championship_algorithm import LCAPolicy
"""

from logic.src.configs.policies.lca import LCAConfig
from .policy_lca import LCAPolicy

__all__ = ["LCAConfig", "LCAPolicy"]
