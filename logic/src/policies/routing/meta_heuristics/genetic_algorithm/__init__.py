"""Genetic Algorithm (GA) solver for VRPP.

Attributes:
    GASolver: Main solver class for the Genetic Algorithm.
    GAPolicy: Policy class for the Genetic Algorithm.
    GAParams: Parameter dataclass for the Genetic Algorithm.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.genetic_algorithm import GAPolicy
    >>> policy = GAPolicy()
"""

from .params import GAParams
from .solver import GASolver

__all__ = ["GASolver", "GAParams"]
