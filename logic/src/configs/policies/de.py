"""
Differential Evolution (DE/rand/1/bin) configuration.

**TRUE DE** replacing the Artificial Bee Colony (ABC) algorithm.

Attributes:
    DEConfig: Attributes for DE configuration.

Example:
    >>> from configs.policies.de import DEConfig
    >>> config = DEConfig()
    >>> config.pop_size
    50
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class DEConfig:
    """Configuration for Differential Evolution policy.

    **Replaces ABC** - Proper DE/rand/1/bin with rigorous mechanics.

    Core DE Components (Storn & Price 1997):
    - Differential mutation: v = x_r1 + F × (x_r2 - x_r3)
    - Binomial crossover with CR parameter
    - Greedy one-to-one selection
    - No metaphor (employed/onlooker/scout bees)
    - No trial counter or abandonment mechanism

    ABC is mathematically isomorphic to DE with fitness-proportionate selection
    instead of greedy selection, but with unnecessary "bee foraging" metaphor.

    DE Advantages over ABC:
    1. Greedy selection is faster than fitness-proportionate
    2. Explicit crossover operator (CR) controls exploration
    3. Simpler algorithm without metaphorical agent types
    4. Proven convergence properties
    5. 25+ years of theoretical foundation

    References:
        Storn, R., & Price, K. (1997). "Differential Evolution – A Simple and
        Efficient Heuristic for Global Optimization over Continuous Spaces."
        Journal of Global Optimization, 11(4), 341-359.

        Karaboga, D. (2005). "An idea based on honey bee swarm for numerical
        optimization." Technical Report TR06.
        [Note: ABC is DE with fitness-proportionate selection]
    Attributes:
        None
    """

    # Population Configuration
    pop_size: int = 50  # Population size (NP)

    # DE Parameters
    mutation_factor: float = 0.8  # Differential weight (F) ∈ [0, 2]
    crossover_rate: float = 0.9  # Crossover probability (CR) ∈ [0, 1]

    # Discrete Mutation Strength
    n_removal: int = 3  # Nodes removed during destroy-repair

    # Runtime Control
    max_iterations: int = 500
    local_search_iterations: int = 100
    evolution_strategy: str = "lamarckian"  # "lamarckian" or "baldwinian"
    time_limit: float = 60.0

    # Infrastructure
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None


    @classmethod
    def from_config(cls, config: Any) -> "DEConfig":
        """Build parameters from a configuration object.

        Args:
            config: Hydra configuration or dictionary.

        Returns:
            An instance of DEConfig.
        """
        return cls(
            pop_size=getattr(config, "pop_size", None),
            mutation_factor=getattr(config, "mutation_factor", 0.8),
            crossover_rate=getattr(config, "crossover_rate", 0.9),
            n_removal=getattr(config, "n_removal", 3),
            max_iterations=getattr(config, "max_iterations", 500),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            evolution_strategy=getattr(config, "evolution_strategy", "lamarckian"),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
