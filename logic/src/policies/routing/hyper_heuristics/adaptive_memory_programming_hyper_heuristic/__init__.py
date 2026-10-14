"""
Package documentation for Adaptive Memory Programming Hyper-Heuristic (AMPHH) module.

This package implements an Adaptive Memory Programming Hyper-Heuristic (AMPHH)
solver that uses adaptive memory to store and reuse high-quality solution
components.

Attributes:
    AMPHHPolicy: AMPHHPolicy class.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics.adaptive_memory_programming_hyper_heuristic import AMPHHPolicy
    >>> solver = AMPHHPolicy()
    >>> solution = solver.solve(problem, multi_day_ctx)
"""

from .policy_amphh import AMPHHPolicy

__all__ = ["AMPHHPolicy"]
