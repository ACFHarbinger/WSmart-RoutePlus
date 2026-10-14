"""
Kernel Search matheuristic module.

This package contains the implementation of the Kernel Search algorithmic framework,
including the simulator adapter, the core Gurobi-based solver, and configuration
schemas.

Reference:
    Angelelli, E., Mansini, R., & Speranza, M. G. (2010). "Kernel Search:
    a new heuristic framework for portfolio selection".

Attributes:
    KernelSearchPolicy: Policy class for Kernel Search matheuristic.

Example:
    >>> from logic.src.policies.routing.matheuristics.kernel_search import KernelSearchPolicy
    >>> policy = KernelSearchPolicy()
"""

from .policy_ks import KernelSearchPolicy

__all__ = ["KernelSearchPolicy"]
