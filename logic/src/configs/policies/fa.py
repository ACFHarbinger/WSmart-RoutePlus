"""
FA (Discrete Firefly Algorithm) configuration for Hydra.

Attributes:
    FAConfig: Configuration for the Discrete Firefly Algorithm policy.

Example:
    >>> from configs.policies.fa import FAConfig
    >>> config = FAConfig()
    >>> config.pop_size
    20
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class FAConfig:
    """
    Configuration for the Discrete Firefly Algorithm policy.

    Attributes:
        pop_size: Number of fireflies.
        beta0: Maximum attractiveness coefficient.
        gamma: Light absorption coefficient.
        alpha_profit: Profit weight in favourability score.
        beta_will: Willingness weight in favourability score.
        gamma_cost: Insertion-cost penalty weight in favourability score.
        alpha_rnd: Random-walk probability per firefly per iteration.
        max_iterations: Maximum iterations.
        n_removal: Number of nodes to remove in each iteration.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        packing: Packing algorithm config list.
        route_improvement: Route improvement operation config list.
    """

    pop_size: int = 20
    beta0: float = 1.0
    gamma: float = 0.1
    alpha_profit: float = 0.5
    beta_will: float = 0.3
    gamma_cost: float = 0.2
    alpha_rnd: float = 0.2
    max_iterations: int = 100
    n_removal: int = 3
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)


    @classmethod
    def from_config(cls, config: Any) -> "FAConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            FAConfig instance.
        """
        return cls(
            pop_size=getattr(config, "pop_size", 20),
            beta0=getattr(config, "beta0", 1.0),
            gamma=getattr(config, "gamma", 0.1),
            alpha_profit=getattr(config, "alpha_profit", 0.5),
            beta_will=getattr(config, "beta_will", 0.3),
            gamma_cost=getattr(config, "gamma_cost", 0.2),
            alpha_rnd=getattr(config, "alpha_rnd", 0.2),
            n_removal=getattr(config, "n_removal", 3),
            max_iterations=getattr(config, "max_iterations", 100),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
