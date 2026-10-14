"""Branch-and-Price Algorithm for VRPP.

This package implements the Branch-and-Price algorithm for the Vehicle Routing
Problem with Profits (VRPP) based on the methodology described in:
Barnhart et al. (1998) "Branch-and-Price: Column Generation for Solving Huge Integer Programs"

The implementation uses:
- Set partitioning master problem formulation
- Column generation via pricing subproblem (resource-constrained shortest path)
- Ryan-Foster branching for binary decisions
- Gurobi for solving restricted master problems

Attributes:
    BranchAndPriceSolver (class): Main optimization search for column generation.
    BranchAndPricePolicy (class): Adapter for the BP solver.
    PricingSubproblem (class): Resource-constrained shortest path subproblem.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price import BranchAndPricePolicy
    >>> policy = BranchAndPricePolicy(config)
    >>> routes, obj = policy.execute(context)
"""

from logic.src.policies.helpers.solvers_and_matheuristics import RCSPPSolver, VRPPMasterProblem

from .bp import BranchAndPriceSolver
from .policy_bp import BranchAndPricePolicy
from .pricing_subproblem import PricingSubproblem

__all__ = [
    "VRPPMasterProblem",
    "PricingSubproblem",
    "RCSPPSolver",
    "BranchAndPriceSolver",
    "BranchAndPricePolicy",
]
