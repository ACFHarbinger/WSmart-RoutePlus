"""
Relax-and-Fix and Optimize (RFO) matheuristic package.

Attributes:
    RelaxFixOptimizePolicy: The RFO policy.

Example:
    >>> from logic.src.policies.routing.matheuristics.relax_and_fix_and_optimize.policy_rfo import RelaxFixOptimizePolicy
    >>> rfo_policy = RelaxFixOptimizePolicy()
    >>> tour, distance, metadata = rfo_policy.execute(**env_state)
"""

from .policy_rfo import RelaxFixOptimizePolicy

__all__ = ["RelaxFixOptimizePolicy"]
