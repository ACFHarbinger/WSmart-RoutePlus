"""Exact Stochastic Dynamic Programming (SDP) Policy.

Implements Backward Induction to solve small SCWCVRP/SIRP instances optimally.

Attributes:
    ExactSDPEngine (class): Core solver engine for ESDP.
    ExactSDPConfig (class): Parameters for the ESDP solver.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.exact_stochastic_dynamic_programming import ExactSDPEngine
"""

from .esdp_engine import ExactSDPEngine
from logic.src.configs.policies.esdp import ExactSDPConfig

__all__ = [
    "ExactSDPConfig",
    "ExactSDPEngine",
]
