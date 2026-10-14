"""
Local Branching with Variable Neighborhood Search (LB-VNS) matheuristic module.

This package implements the LB-VNS algorithm, a hybrid optimization approach
that leverages the exploration capabilities of Variable Neighborhood Search (VNS)
to escape local optima, while using Local Branching (LB) for rigorous
intensification within the mathematical programming search space.

Based on:
    Hansen, P., Mladenović, N., & Urošević, D. (2006).
    "Variable neighborhood search and local branching".
    Computers & Operations Research.

Attributes:
    LocalBranchingVNSPolicy (LocalBranchingVNSPolicy): Implementation of the LB-VNS matheuristic.

Example:
    >>> from logic.src.policies.routing.matheuristics.local_branching_variable_neighborhood_search import LocalBranchingVNSPolicy
    >>> policy = LocalBranchingVNSPolicy()
    >>> solution = policy.solve(
    ...     dist_matrix=dist_matrix,
    ...     wastes=wastes,
    ...     capacity=capacity,
    ...     R=revenue,
    ...     C=cost_unit,
    ...     mandatory_nodes=mandatory_nodes,
    ... )
    >>> print(solution["tour"])
"""

from .policy_lb_vns import LocalBranchingVNSPolicy

__all__ = ["LocalBranchingVNSPolicy"]
