"""Configuration parameters for the Discrete Firefly Algorithm (FA) solver.

Attributes:
    FAParams: Parameter dataclass for the Firefly Algorithm.

Example:
    >>> params = FAParams(pop_size=10)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class FAParams:
    """Configuration parameters for the Discrete Firefly Algorithm.

    Firefly brightness = net profit. Attractiveness decays with the
    discrete routing distance.

    Attributes:
        pop_size: Number of fireflies.
        beta0: Maximum attractiveness (at distance 0).
        gamma: Light absorption coefficient controlling distance decay.
        alpha_profit: Weight of node profit in favourability score.
        beta_will: Weight of node willingness (waste fill fraction).
        gamma_cost: Weight of insertion cost in score (penalty).
        alpha_rnd: Probability of random-walk perturbation.
        n_removal: Number of nodes to remove during random walk.
        max_iterations: Maximum algorithm iterations.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        seed: Random seed.
        vrpp: Whether solving VRP with Profits.
        profit_aware_operators: Whether to use profit-aware operators.
    """

    pop_size: int = 20
    beta0: float = 1.0
    gamma: float = 0.1
    alpha_profit: float = 0.5
    beta_will: float = 0.3
    gamma_cost: float = 0.2
    alpha_rnd: float = 0.2
    n_removal: int = 3
    max_iterations: int = 100
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "FAParams":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            FAParams instance.
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
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
