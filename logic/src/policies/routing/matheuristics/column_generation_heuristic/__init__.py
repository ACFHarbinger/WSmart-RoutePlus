"""
Column Generation Heuristic (CGH) package.

Attributes:
    ColumnGenerationHeuristicPolicy: Policy class for CGH-based routing.

Example:
    >>> from logic.src.policies.routing.matheuristics.column_generation_heuristic import ColumnGenerationHeuristicPolicy
    >>> policy = ColumnGenerationHeuristicPolicy()
"""

from .policy_cgh import ColumnGenerationHeuristicPolicy

__all__ = ["ColumnGenerationHeuristicPolicy"]
