"""
Hyper-Heuristic Ant Colony Optimization.

This package implements a hyper-heuristic approach where ants optimize the
sequence of local search operators applied to a solution, rather than
constructing the solution itself.

Attributes:
    HYPER_OPERATORS (list): List of available hyper-heuristic operators.
    HyperHeuristicACO (class): The main solver class.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic import HyperHeuristicACO
    >>> solver = HyperHeuristicACO(dist_matrix, wastes, capacity, R, C, cfg)
"""

from .hyper_aco import HyperHeuristicACO
from .hyper_operators import HYPER_OPERATORS, OPERATOR_NAMES

__all__ = ["HyperHeuristicACO", "HYPER_OPERATORS", "OPERATOR_NAMES"]
