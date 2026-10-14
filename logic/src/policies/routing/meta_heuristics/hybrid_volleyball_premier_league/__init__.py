"""
Hybrid Volleyball Premier League (HVPL) algorithm.

Integrates ACO (intelligent initialization), VPL + HGS (population evolution),
and ALNS (deep local search) for solving complex routing problems.

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023). "Volleyball premier league
    algorithm with ACO and ALNS for simultaneous pickup delivery location
    routing problem."

Attributes:
    HVPLParams: Parameters for the HVPL solver.
    HVPLPolicy: Policy adapter for HVPL.
    HVPLSolver: Core HVPL solver logic.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.hybrid_volleyball_premier_league import HVPLSolver
"""

from .params import HVPLParams
from .policy_hvpl import HVPLPolicy
from .solver import HVPLSolver

__all__ = ["HVPLParams", "HVPLSolver", "HVPLPolicy"]
