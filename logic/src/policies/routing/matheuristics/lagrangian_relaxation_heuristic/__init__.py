"""
Lagrangian Relaxation Heuristic (LRH) package.

Attributes:
    LagrangianRelaxationHeuristicPolicy: Policy class for LRH-based routing.

Example:
    >>> from logic.src.policies.routing.matheuristics.lagrangian_relaxation_heuristic import LagrangianRelaxationHeuristicPolicy
    >>> policy = LagrangianRelaxationHeuristicPolicy()
"""

from .policy_lrh import LagrangianRelaxationHeuristicPolicy

__all__ = ["LagrangianRelaxationHeuristicPolicy"]
