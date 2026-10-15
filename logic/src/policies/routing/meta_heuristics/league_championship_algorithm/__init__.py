"""
League Championship Algorithm (LCA) for VRPP.

LCA is a population-based metaheuristic inspired by the competition in a
sports league.

Attributes:
    LCAConfig: Configuration parameters for LCA.
    LCAPolicy: Policy adapter for LCA.

"""

from logic.src.configs.policies.lca import LCAConfig

from .policy_lca import LCAPolicy

__all__ = ["LCAConfig", "LCAPolicy"]
