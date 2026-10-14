"""
Configuration parameters for Memetic Island Model Genetic Algorithm.

Attributes:
    MemeticAlgorithmIslandModelParams: Parameters for the MAIM solver.

Example:
    >>> params = MemeticAlgorithmIslandModelParams(n_islands=5, island_size=4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MemeticAlgorithmIslandModelParams:
    """
    Parameters for Memetic Algorithm with Island Model (MA-IM).

    Attributes:
        n_islands: Number of sub-populations (islands).
        island_size: Individuals per sub-population.
        max_generations: Evolution cycles.
        stagnation_limit: Generations without improvement before island regeneration.
        n_removal: Nodes removed per mutation/perturbation step.
        local_search_iterations: Number of local search iterations.
        time_limit: Wall-clock time limit in seconds.
        vrpp: Whether to solve as a VRP with profits.
        profit_aware_operators: Whether to use profit-aware heuristics.
        seed: Random seed for reproducibility.
    """

    n_islands: int = 5  # K islands (analogous to teams)
    island_size: int = 4  # N individuals per island
    max_generations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    local_search_iterations: int = 100
    time_limit: float = 60.0
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "MemeticAlgorithmIslandModelParams":
        """Create parameters from a configuration object.

        Args:
            config: Configuration source (dataclass or object).

        Returns:
            MemeticAlgorithmIslandModelParams: Initialized runtime parameters.
        """
        return cls(
            n_islands=getattr(config, "n_islands", 5),
            island_size=getattr(config, "island_size", 4),
            max_generations=getattr(config, "max_generations", 50),
            stagnation_limit=getattr(config, "stagnation_limit", 5),
            n_removal=getattr(config, "n_removal", 1),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", 42),
        )
