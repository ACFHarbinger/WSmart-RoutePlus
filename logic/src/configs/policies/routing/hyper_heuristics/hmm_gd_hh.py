"""
HMM-GD (Hidden Markov Model + Great Deluge) configuration for Hydra.

Attributes:
    HMMGDHHConfig: Configuration for the HMM-GD (Hidden Markov Model + Great Deluge) policy.

Example:
    >>> from configs.policies.hmm_gd_hh import HMMGDHHConfig
    >>> config = HMMGDHHConfig()
    >>> config.max_iterations
    500
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class HMMGDHHConfig:
    """
    Configuration for the Hidden Markov Model + Great Deluge Hyper-Heuristic policy.

    Attributes:
        max_iterations: Total LLH applications.
        flood_margin: Initial water level offset as fraction of initial profit.
        rain_speed: Rate of water level decrease per iteration.
        learning_rate: Online HMM transition probability update step.
        n_removal: Nodes removed per destroy step.
        n_llh: LLH pool size (fixed at 5).
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        packing: Packing algorithm config list.
        route_improvement: Route improvement operation config list.
    """

    max_iterations: int = 500
    flood_margin: float = 0.05
    rain_speed: float = 0.001
    learning_rate: float = 0.1
    n_removal: int = 2
    n_llh: int = 5
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)


    @classmethod
    def from_config(cls, config: Any) -> "HMMGDHHConfig":
        """
        Create parameters from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            HMMGDHHConfig: Instance of HMMGDHHConfig.
        """
        return cls(
            max_iterations=getattr(config, "max_iterations", 500),
            flood_margin=getattr(config, "flood_margin", 0.05),
            rain_speed=getattr(config, "rain_speed", 0.001),
            learning_rate=getattr(config, "learning_rate", 0.1),
            n_removal=getattr(config, "n_removal", 2),
            n_llh=getattr(config, "n_llh", 5),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
