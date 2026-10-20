"""
Pure Island Model GA configuration.

Replaces SLC with rigorous GA terminology.

Attributes:
    MemeticAlgorithmIslandModelConfig: Configuration for the Memetic Algorithm Island Model policy.

Example:
    >>> from configs.policies.ma_im import MemeticAlgorithmIslandModelConfig
    >>> config = MemeticAlgorithmIslandModelConfig()
    >>> config.max_generations
    50
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.route_improvement import RouteImprovingConfig


@dataclass
class MemeticAlgorithmIslandModelConfig:
    """Configuration for Memetic Algorithm Island Model (MA-IM) policy.

    Functionally equivalent to SLC.

    Attributes:
        n_islands (int): Number of islands in the GA.
        island_size (int): Number of individuals per island.
        max_generations (int): Maximum number of generations to run.
        stagnation_limit (int): Number of generations without improvement before triggering stagnation handling.
        n_removal (int): Number of individuals to remove during replacement.
        local_search_iterations (int): Number of local search iterations per individual.
        time_limit (float): Time limit in seconds.
        seed (Optional[int]): Random seed.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        packing (Optional[List[PackingConfig]]): Mandatory customers/requests selection strategies.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement strategies.
    """

    n_islands: int = 5
    island_size: int = 4
    max_generations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    @classmethod
    def from_config(cls, config: Any) -> "MemeticAlgorithmIslandModelConfig":
        """Create parameters from a configuration object.

        Args:
            config: Configuration source (dataclass or object).

        Returns:
            MemeticAlgorithmIslandModelConfig: Initialized runtime parameters.
        """
        return cls(
            n_islands=getattr(config, "n_islands", 5),
            island_size=getattr(config, "island_size", 4),
            max_generations=getattr(config, "max_generations", 50),
            stagnation_limit=getattr(config, "stagnation_limit", 5),
            n_removal=getattr(config, "n_removal", 1),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", 42),
        )
