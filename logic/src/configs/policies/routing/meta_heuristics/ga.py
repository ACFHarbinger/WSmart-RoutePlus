"""
GA (Genetic Algorithm) configuration for Hydra.

Attributes:
    GAConfig: Configuration for the Genetic Algorithm policy.

Example:
    >>> from configs.policies.ga import GAConfig
    >>> config = GAConfig()
    >>> config.pop_size
    20
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class GAConfig:
    """Configuration for the Genetic Algorithm policy.

    Attributes:
        pop_size (int): Number of individuals in the population.
        max_generations (int): Maximum number of generations.
        crossover_rate (float): Probability of crossover.
        mutation_rate (float): Probability of mutation.
        tournament_size (int): Size of the tournament selection.
        n_removal (int): Number of routes to remove.
        time_limit (float): Time limit in seconds.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        seed (Optional[int]): Seed for the random number generator.
        packing (Optional[List[PackingConfig]]): Packing configurations.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement configurations.
    """

    pop_size: int = 30
    max_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 3
    n_removal: int = 2
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)


    @classmethod
    def from_config(cls, config: Any) -> "GAConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            GAConfig instance.
        """
        return cls(
            pop_size=getattr(config, "pop_size", 30),
            max_generations=getattr(config, "max_generations", 100),
            crossover_rate=getattr(config, "crossover_rate", 0.8),
            mutation_rate=getattr(config, "mutation_rate", 0.1),
            tournament_size=getattr(config, "tournament_size", 3),
            n_removal=getattr(config, "n_removal", 2),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
