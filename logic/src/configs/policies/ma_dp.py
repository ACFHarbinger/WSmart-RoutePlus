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


    @classmethod
    def from_config(cls, config: Any) -> "MemeticAlgorithmDualPopulationConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration source (dataclass or object).

        Returns:
            MemeticAlgorithmDualPopulationConfig: Initialized runtime parameters.
        """
        return cls(
            population_size=getattr(config, "population_size", 30),
            max_iterations=getattr(config, "max_iterations", 200),
            diversity_injection_rate=getattr(config, "diversity_injection_rate", 0.2),
            elite_learning_weights=getattr(config, "elite_learning_weights", None),
            elite_count=getattr(config, "elite_count", 3),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 300.0),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )

    def __post_init__(self):
        """Validate parameter constraints and set default elite weights.

        Args:
            None.

        Returns:
            None.
        """
        assert self.population_size > 0, "population_size must be positive"
        assert 0 <= self.diversity_injection_rate <= 1, "diversity_injection_rate must be in [0, 1]"
        assert self.elite_count >= 3, "elite_count must be at least 3 for elite-guided construction"

        # Set default elite learning weights if not provided
        if self.elite_learning_weights is None:
            # Default: Best gets 0.5, second-best gets 0.3, third-best gets 0.2
            self.elite_learning_weights = [0.5, 0.3, 0.2]

        # Ensure we have enough weights for elite_count
        if len(self.elite_learning_weights) < self.elite_count:
            raise ValueError(
                f"elite_learning_weights must have at least {self.elite_count} values, "
                f"got {len(self.elite_learning_weights)}"
            )

        # Validate elite learning weights sum to 1.0
        total_weight = sum(self.elite_learning_weights[: self.elite_count])
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Elite learning weights (first {self.elite_count}) must sum to 1.0, got {total_weight}")

    @property
    def substitution_rate(self) -> float:
        """Alias for diversity_injection_rate to match VPL exactly.

        Args:
            None.

        Returns:
            float: The diversity injection rate.
        """
        return self.diversity_injection_rate

    @property
    def elite_size(self) -> int:
        """Alias for elite_count to match VPL exactly.

        Args:
            None.

        Returns:
            int: The elite count.
        """
        return self.elite_count

    @property
    def coaching_weight_1(self) -> float:
        """Alias for first elite learning weight to match VPL exactly.

        Args:
            None.

        Returns:
            float: The first coaching weight.
        """
        assert self.elite_learning_weights is not None and len(self.elite_learning_weights) >= 1
        return self.elite_learning_weights[0]

    @property
    def coaching_weight_2(self) -> float:
        """Alias for second elite learning weight to match VPL exactly.

        Args:
            None.

        Returns:
            float: The second coaching weight.
        """
        assert self.elite_learning_weights is not None and len(self.elite_learning_weights) >= 2
        return self.elite_learning_weights[1]

    @property
    def coaching_weight_3(self) -> float:
        """Alias for third elite learning weight to match VPL exactly.

        Args:
            None.

        Returns:
            float: The third coaching weight.
        """
        assert self.elite_learning_weights is not None and len(self.elite_learning_weights) >= 3
        return self.elite_learning_weights[2]
