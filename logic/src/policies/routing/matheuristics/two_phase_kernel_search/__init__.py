"""
Two-Phase Kernel Search (TPKS) matheuristic package.

Attributes:
    TPKSPolicy: The Two-Phase Kernel Search policy.

Example:
    >>> from logic.src.policies.routing.matheuristics.two_phase_kernel_search import TPKSPolicy
    >>> tpks_policy = TPKSPolicy()
    >>> tpks_policy.execute(distance_matrix=..., wastes={}, capacity=1e9, R=1.0, C=1.0, mandatory_nodes=[])
"""

from .policy_tpks import TPKSPolicy

__all__ = ["TPKSPolicy"]
