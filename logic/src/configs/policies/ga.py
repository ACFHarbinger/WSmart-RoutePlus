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
