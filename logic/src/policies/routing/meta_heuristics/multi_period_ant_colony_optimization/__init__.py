"""
Multi-Period Ant Colony Optimization (MP-ACO) matheuristic package.

Provides a multi-period ant colony optimization approach to solve complex
vehicle routing problems with temporal constraints.

Attributes:
    MultiPeriodACOPolicy: Policy class for the MP-ACO approach.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.multi_period_ant_colony_optimization import MultiPeriodACOPolicy
"""

from .policy_mp_aco import MultiPeriodACOPolicy

__all__ = ["MultiPeriodACOPolicy"]
