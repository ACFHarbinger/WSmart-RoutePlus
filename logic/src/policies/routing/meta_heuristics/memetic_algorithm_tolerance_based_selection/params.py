"""
Configuration parameters for Stochastic Tournament Genetic Algorithm (STGA).

Attributes:
    MemeticAlgorithmToleranceBasedSelectionParams: Parameters for the MATBS solver.

Example:
    >>> params = MemeticAlgorithmToleranceBasedSelectionParams(population_size=10)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MemeticAlgorithmToleranceBasedSelectionParams:
    """
    Parameters for Memetic Algorithm with Tolerance-based Selection (MA-TS).

    Attributes:
        seed: Random seed for reproducibility.
        population_size: Number of candidate solutions (islands).
        max_iterations: Maximum number of evolution cycles.
        tolerance_pct: Infeasibility tolerance as fraction of average fitness.
        recombination_rate: Probability of crossover vs mutation.
        perturbation_strength: Number of nodes to remove in mutation.
        n_removal: Alternative name for perturbation strength.
        local_search_iterations: Local search refinement iterations.
        vrpp: Whether to solve as a VRP with profits.
        profit_aware_operators: Whether to use profit-aware heuristics.
        time_limit: Wall-clock time limit in seconds (0 = no limit).
    """

    seed: Optional[int] = None

    # Population structure (LCA: n_teams)
    population_size: int = 10

    # Evolution control (LCA: max_iterations)
    max_iterations: int = 100

    # Infeasibility tolerance (LCA: tolerance_pct)
    # CRITICAL FEATURE: Allows diversity preservation
    tolerance_pct: float = 0.05  # 5% tolerance for similar solutions

    # Genetic operators (LCA: crossover_prob, n_removal)
    recombination_rate: float = 0.6  # Probability of crossover vs mutation
    perturbation_strength: int = 2  # Nodes removed in mutation
    n_removal: int = 1

    # Local search refinement
    local_search_iterations: int = 100

    # Profit-awareness flags
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Resource constraints
    time_limit: float = 60.0

    @classmethod
    def from_config(cls, config: Any) -> "MemeticAlgorithmToleranceBasedSelectionParams":
        """Create parameters from a configuration object.

        Args:
            config: Configuration source (dataclass or object).

        Returns:
            MemeticAlgorithmToleranceBasedSelectionParams: Initialized runtime parameters.
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
            vrpp=getattr(config, "vrpp", True),
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
