"""Exact Stochastic Dynamic Programming (SDP) Policy.

Implements Backward Induction to solve small SCWCVRP/SIRP instances optimally.

Attributes:
    ExactSDPEngine (class): Core solver engine for ESDP.
    SDPParams (class): Parameters for the ESDP solver.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming import ExactSDPEngine
"""

from .esdp_engine import ExactSDPEngine
from .params import SDPParams

__all__ = [
    "SDPParams",
    "ExactSDPEngine",
]
