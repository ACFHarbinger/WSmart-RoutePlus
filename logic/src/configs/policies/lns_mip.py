"""LnsMipConfig: configuration dataclass for the ``lns_mip`` policy.

Moved from the policy's ``params.py`` by the WP-06 scripted migration(Q-8b): the config dataclass is the single source of truth (P-10).
"""

from __future__ import annotations

from dataclasses import dataclass

"""
Parameters for Large Neighborhood Search Mixed Integer Programming.

Attributes:
    LNSMIPParams: Dataclass holding LNS-MIP solver hyperparameters.

Example:
    >>> from logic.src.policies.routing.matheuristics.large_neighborhood_search_mixed_integer_programming.configs.policies.lns_mip import LnsMipConfig
    >>> params = LNSMIPParams(k_destroy=10, max_iterations=200)
"""



@dataclass
class LnsMipConfig:
    """
    Parameters for the LNSMIP algorithm.

    Attributes:
        k_destroy: Number of nodes to destroy in each iteration.
        d_destroy: Number of neighbors to destroy at each step.
        max_iterations: Maximum number of iterations.
        mip_time_limit: Time limit for MIP solves.
        mip_gap: MIP relative gap tolerance.
        acceptance: Acceptance strategy ("improving" or "sa").
        sa_temperature: Initial temperature for Simulated Annealing.
        sa_cooling: Cooling rate for Simulated Annealing.
        seed: Random seed for reproducibility.
    """

    k_destroy: int = 10
    d_destroy: int = 3
    max_iterations: int = 200
    mip_time_limit: float = 10.0
    mip_gap: float = 0.01
    acceptance: str = "improving"  # "improving" | "sa"
    sa_temperature: float = 100.0
    sa_cooling: float = 0.97
    seed: int = 42
