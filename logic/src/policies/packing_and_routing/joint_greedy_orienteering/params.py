"""Joint Greedy Orienteering Parameters.

This module defines the configuration parameters for the randomized greedy
orienteering policy.

Attributes:
    JointGreedyParams: Data class for JGO configuration.

Example:
    >>> params = JointGreedyParams(k_best=5, n_starts=20)
    >>> print(params.k_best)
    5
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class JointGreedyParams:
    """Configuration parameters for the Joint Greedy Orienteering solver.

    Attributes:
        k_best (int): Number of top candidates to sample from (RCL size).
        n_starts (int): Number of multi-start randomized greedy iterations.
        distance_weight (float): Weight for distance in the scoring metric.
        seed (int): Random seed for reproducibility.
        time_limit (float): Maximum allowed execution time in seconds.
    """

    # --- Restricted Candidate List (RCL) ---
    k_best: int = 3  # Number of top candidates to sample from (randomized greedy)
    n_starts: int = 10  # Number of multi-start iterations

    # --- Objective Weights ---
    # metric = (revenue - distance_cost) / distance_growth
    distance_weight: float = 1.0

    # --- Infrastructure ---
    seed: int = 42
    time_limit: float = 30.0

    @classmethod
    def from_config(cls, config: Any) -> "JointGreedyParams":
        """Build params from a config dataclass or dictionary.

        Args:
            config (Any): Configuration source (dict, Hydra config, etc.).

        Returns:
            JointGreedyParams: The initialized parameters.
        """
        if config is None:
            return cls()

        kwargs = {}
        # Handle both dicts and dataclasses (Hydra/OmegaConf)
        if hasattr(config, "__dict__"):
            source = config.__dict__
        elif hasattr(config, "items"):
            source = config
        else:
            return cls()

        for field in cls.__dataclass_fields__:
            if field in source and source[field] is not None:
                kwargs[field] = source[field]

        return cls(**kwargs)
