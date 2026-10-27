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
from typing import Any, List, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig
from logic.src.configs.policies.route_improvement import RouteImprovingConfig


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

    @classmethod
    def from_config(cls, config: Any) -> "MemeticAlgorithmToleranceBasedSelectionConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration source (dataclass or object).

        Returns:
            MemeticAlgorithmToleranceBasedSelectionConfig: Initialized runtime parameters.
        """
        return cls(
            population_size=getattr(config, "population_size", 10),
            max_iterations=getattr(config, "max_iterations", 100),
            tolerance_pct=getattr(config, "tolerance_pct", 0.05),
            recombination_rate=getattr(config, "recombination_rate", 0.6),
            perturbation_strength=getattr(config, "perturbation_strength", 2),
            n_removal=getattr(config, "n_removal", 1),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", 42),
        )

    def __post_init__(self):
        """Validate parameter constraints.

        Args:
            None.

        Returns:
            None.
        """
        assert self.population_size > 0, "population_size must be positive"
        assert self.population_size % 2 == 0, "population_size must be even for pairwise matching"
        assert 0 <= self.tolerance_pct <= 1, "tolerance_pct must be in [0, 1]"
        assert 0 <= self.recombination_rate <= 1, "recombination_rate must be in [0, 1]"
        assert self.perturbation_strength > 0, "perturbation_strength must be positive"
        assert self.n_removal > 0, "n_removal must be positive"
