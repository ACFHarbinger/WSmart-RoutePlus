"""
Relaxation Enforced Neighborhood Search (RENS) Policy Module.

This module provides the implementation of the RENS matheuristic, a
mathematical programming-based start heuristic designed for Mixed-Integer
Programming (MIP) problems like VRPP and WCVRP.

Key Components:
    - RENSSolver: Core class-based engine that implements the LP relaxation,
      variable fixing, and sub-MIP solve phases.
    - RENSPolicy: Simulator adapter that integrates the solver into the
      high-level routing policy workflow.
    - run_rens_gurobi: functional wrapper for quick optimization calls.

Attributes:
    RENSPolicy: The RENS policy.
    run_rens_gurobi: The RENS solver wrapper.

Example:
    >>> from logic.src.policies.routing.matheuristics.relaxation_enforced_neighborhood_search.policy_rens import RENSPolicy
    >>> rens_policy = RENSPolicy()
    >>> tour, distance, metadata = rens_policy.execute(**env_state)
"""

from .policy_rens import RENSPolicy
from .solver import run_rens_gurobi

__all__ = ["RENSPolicy", "run_rens_gurobi"]
