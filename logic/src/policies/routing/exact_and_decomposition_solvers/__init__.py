"""Exact and decomposition-based solvers for the VRPP.

Attributes:
    branch_and_bound (module): Classic exact solvers using tree search.
    branch_and_cut (module): Cutting plane based exact solvers.
    branch_and_price (module): Column generation based exact solvers.
    branch_and_price_and_cut (module): Unified BPC engine.
    constraint_programming_with_boolean_satisfiability (module): CP-SAT solvers.
    exact_stochastic_dynamic_programming (module): DP solvers for stochastic VRPP.
    integer_l_shaped_benders_decomposition (module): Benders decomposition for stochastic VRPP.
    logic_based_benders_decomposition (module): Logic-based Benders for multi-period VRPP.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound import BranchAndBoundPolicy
    >>> solver = BranchAndBoundPolicy(config)
    >>> routes, obj = solver.execute(context)
"""

from . import (
    branch_and_bound,
    branch_and_cut,
    branch_and_price,
    branch_and_price_and_cut,
)

__all__ = [
    "branch_and_bound",
    "branch_and_cut",
    "branch_and_price",
    "branch_and_price_and_cut",
]
