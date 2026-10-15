"""
ABC (Artificial Bee Colony) configuration for Hydra.

Attributes:
    ABCConfig: ABC policy configuration.

Example:
    >>> from logic.src.configs.policies import ABCConfig
    >>> config = ABCConfig()
    >>> print(config)
    ABCConfig(n_sources=20, limit=10, max_iterations=200, n_removal=1, local_search_iterations=500, time_limit=60.0, seed=None,  profit_aware_operators=False, packing=[], route_improvement=[])
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ABCConfig:
    """
    Configuration for the Artificial Bee Colony policy.

    Attributes:
        n_sources: Number of food sources (employed bees).
        limit: Abandonment threshold for scout bee phase.
        max_iterations: Maximum ABC cycles.
        n_removal: Nodes removed per neighbourhood perturbation.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        packing: Packing algorithm config list.
        route_improvement: Route improvement operation config list.
    """

    n_sources: int = 20
    limit: int = 10
    max_iterations: int = 200
    n_removal: int = 1
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)


    @classmethod
    def from_config(cls, config: Any) -> "ABCConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object (Hydra config or DictConfig).

        Returns:
            An instance of ABCConfig.
        """
        return cls(
            n_sources=max(1, getattr(config, "n_sources", 20)),
            limit=max(1, getattr(config, "limit", 10)),
            max_iterations=max(1, getattr(config, "max_iterations", 200)),
            n_removal=max(1, getattr(config, "n_removal", 1)),
            local_search_iterations=max(0, getattr(config, "local_search_iterations", 100)),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
