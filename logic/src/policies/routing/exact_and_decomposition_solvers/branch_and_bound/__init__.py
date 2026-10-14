"""Branch-and-Bound Policy Module.

This package implements the foundational Branch-and-Bound (BB) algorithm for exact
combinatorial optimization, specifically specialized for the Vehicle Routing
Problem with Profits (VRPP) and Capacitated Waste Collection VRP (CWC VRP).

Implementation follows the methodology proposed by Land and Doig (1960), utilizing
Linear Programming (LP) relaxations to provide rigorous mathematical bounds during
tree exploration.

Attributes:
    BranchAndBoundPolicy (class): Main policy adapter for BB solvers.
    run_bb_optimizer (function): Unified dispatcher for BB formulations.
    run_bb_mtz (function): Miller-Tucker-Zemlin compact solver.
    run_bb_dfj (function): Dantzig-Fulkerson-Johnson lazy cut solver.
    run_bb_lr_uop (function): Lagrangian Relaxation with uncapacitated OP bounding.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_bound import run_bb_optimizer
    >>> routes, obj = run_bb_optimizer(dist_matrix, wastes, capacity, R, C)
"""

from .dfj import run_bb_dfj
from .dispatcher import run_bb_optimizer
from .lr_uop import run_bb_lr_uop
from .mtz import run_bb_mtz
from .policy_bb import BranchAndBoundPolicy

__all__ = [
    "BranchAndBoundPolicy",
    "run_bb_optimizer",
    "run_bb_mtz",
    "run_bb_dfj",
    "run_bb_lr_uop",
]
