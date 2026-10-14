"""
Selection Hyper-Heuristic (SHH) package.

Attributes:
    SelectionHHPolicy: Solver class that solves VRPP using Selection Hyper-Heuristic.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics.selection_hyper_heuristic import SelectionHHPolicy
    >>> solver = SelectionHHPolicy(config)
    >>> best_solution, best_profit, best_cost = solver.solve()
    >>> print(best_solution, best_profit, best_cost)
"""

from .policy_shh import SelectionHHPolicy

__all__ = ["SelectionHHPolicy"]
