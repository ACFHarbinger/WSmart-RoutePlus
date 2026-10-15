"""
Island Model Genetic Algorithm with Stochastic Tournaments (IMGA-ST) Configuration.

Attributes:
    MemeticAlgorithmDualPopulationConfig: Configuration for the Memetic Algorithm Dual Population policy.

Example:
    >>> from configs.policies.ma_dp import MemeticAlgorithmDualPopulationConfig
    >>> config = MemeticAlgorithmDualPopulationConfig()
    >>> config.population_size
    30
    >>> config.time_limit
    300.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class MemeticAlgorithmDualPopulationConfig:
    """
    Configuration for the Memetic Algorithm Dual Population policy.

    Attributes:
        population_size (int): Size of the population.
        max_iterations (int): Maximum number of iterations.
        diversity_injection_rate (float): Rate of diversity injection.
        elite_learning_weights (Optional[List[float]]): Weights for elite learning.
        elite_count (int): Number of elites.
        local_search_iterations (int): Number of local search iterations.
        time_limit (float): Time limit in seconds.
        seed (Optional[int]): Random seed.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        packing (List[str]): Mandatory customer/request selection strategy.
        route_improvement (List[Any]): Route improvement strategies.
    """

    # MADP Parameters
    population_size: int = 30
    max_iterations: int = 200
    diversity_injection_rate: float = 0.2
    elite_learning_weights: Optional[List[float]] = None
    elite_count: int = 3

    # Operators
    local_search_iterations: int = 500
    time_limit: float = 300.0
    seed: Optional[int] = None

    # Common policy fields
    profit_aware_operators: bool = False
    packing: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
