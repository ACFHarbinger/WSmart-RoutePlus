"""
Configuration for the Memetic Algorithm Tolerance-based Selection (MA-TS) policy.

Attributes:
    MemeticAlgorithmToleranceBasedSelectionConfig: Configuration for the Memetic Algorithm Tolerance-based Selection (MA-TS) policy.

Example:
    >>> from configs.policies.ma_ts import MemeticAlgorithmToleranceBasedSelectionConfig
    >>> config = MemeticAlgorithmToleranceBasedSelectionConfig()
    >>> config.time_limit
    60.0
    >>> config.max_iterations
    100
    >>> config.tolerance_pct
    0.05
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MemeticAlgorithmToleranceBasedSelectionConfig:
    """Configuration for Memetic Algorithm Tolerance-based Selection (MA-TS) policy.

    GA with sigmoid-based pairwise tournament selection.

    Attributes:
        population_size (int): Size of the population.
        max_iterations (int): Maximum number of iterations.
        tolerance_pct (float): Tolerance percentage for pairwise tournament selection.
        recombination_rate (float): Probability of applying recombination.
        perturbation_strength (int): Strength of perturbation.
        n_removal (int): Number of individuals to remove during replacement.
        local_search_iterations (int): Number of local search iterations per individual.
        time_limit (float): Time limit in seconds.
        seed (Optional[int]): Random seed.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        packing (Optional[List[PackingConfig]]): Mandatory customers/requests selection strategies.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement strategies.
    """

    population_size: int = 10
    max_iterations: int = 100
    tolerance_pct: float = 0.05
    recombination_rate: float = 0.6
    perturbation_strength: int = 2
    n_removal: int = 1
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
