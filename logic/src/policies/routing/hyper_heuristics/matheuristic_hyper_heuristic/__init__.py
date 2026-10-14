"""
Init documentation for Matheuristic Hyper-Heuristic.

Exposes the MHHPolicy class for use in routing.

Attributes:
    MHHPolicy: Adapter for the MHH solver.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics import MHHPolicy
    >>> policy = MHHPolicy()
"""

from .policy_mhh import MHHPolicy

__all__ = ["MHHPolicy"]
