"""
Slack Induction by String Removal (SISR) Package.

This package implements the SISR heuristic for the Vehicle Routing Problem.
SISR is an iterated local search method that destroys routes by removing strings
of nodes and repairs them using a greedy heuristic with blinks.

Attributes:
    SISRConfig (class): Configuration parameters.
    SISRSolver (class): Main solver class.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.slack_induction_by_string_removal.solver import SISRSolver
    >>> solver = SISRSolver()
    >>> routes, cost, revenue = solver.solve()
"""

from logic.src.configs.policies.sisr import SISRConfig

from .solver import SISRSolver

__all__ = ["SISRConfig", "SISRSolver"]
