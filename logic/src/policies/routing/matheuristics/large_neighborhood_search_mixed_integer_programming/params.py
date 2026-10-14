"""
Parameters for Large Neighborhood Search Mixed Integer Programming.

Attributes:
    LNSMIPParams: Dataclass holding LNS-MIP solver hyperparameters.

Example:
    >>> from logic.src.policies.routing.matheuristics.large_neighborhood_search_mixed_integer_programming.params import LNSMIPParams
    >>> params = LNSMIPParams(k_destroy=10, max_iterations=200)
"""

from dataclasses import MISSING, dataclass, fields
from typing import Any


@dataclass
class LNSMIPParams:
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

    @classmethod
    def from_config(cls, config: Any) -> "LNSMIPParams":
        """
        Creates an LNSMIPParams instance from a configuration object.

        Args:
            config: Configuration object containing parameters.

        Returns:
            An LNSMIPParams instance.
        """
        params = {}
        for f in fields(cls):
            val = getattr(config, f.name, f.default)
            # Handle MISSING from dataclasses fields or OmegaConf
            if str(val) == "MISSING" or val is MISSING:
                continue
            params[f.name] = val
        return cls(**params)
