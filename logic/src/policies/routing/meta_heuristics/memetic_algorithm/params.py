"""Configuration parameters for the Memetic Algorithm (MA) solver.

Attributes:
    MAParams: Parameter dataclass for the Memetic Algorithm.

Example:
    >>> params = MAParams(pop_size=50)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MAParams:
    """Hyper-parameters for the Memetic Algorithm (MA) solver.

    Attributes:
        pop_size: The number of active individuals in each generation.
        max_generations: The maximum number of generational iterations.
        crossover_rate: Probability parents exchange information.
        mutation_rate: Probability of structural perturbation.
        local_search_rate: Probability intensive local search is applied.
        tournament_size: Candidates selected for competitive selection.
        n_removal: Nodes removed/re-inserted during mutation.
        time_limit: Maximum duration allowed for the search.
        vrpp: Whether solving VRP with Profits.
        profit_aware_operators: Whether to use profit-aware operators.
        seed: Random seed for reproducibility.
    """

    # Population and Iteration Settings
    pop_size: int = 30
    max_generations: int = 100

    # Probability Coefficients for Fig. 3.2 Pipeline
    crossover_rate: float = 0.8  # Probability of information exchange
    mutation_rate: float = 0.1  # Probability of random perturbation
    local_search_rate: float = 1.0  # Intensive search probability

    # Selection and Operator Scale
    tournament_size: int = 3  # Competitive selection pressure
    n_removal: int = 2  # Mutation aggressiveness

    # Resource Guard
    time_limit: float = 60.0  # Wall-clock timeout (seconds)
    vrpp: bool = True
    profit_aware_operators: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "MAParams":
        """Create parameters from a configuration object.

        Args:
            config: Configuration source.

        Returns:
            Instantiated MAParams.
        """
        return cls(
            pop_size=getattr(config, "pop_size", 30),
            max_generations=getattr(config, "max_generations", 100),
            crossover_rate=getattr(config, "crossover_rate", 0.8),
            mutation_rate=getattr(config, "mutation_rate", 0.1),
            local_search_rate=getattr(config, "local_search_rate", 1.0),
            tournament_size=getattr(config, "tournament_size", 3),
            n_removal=getattr(config, "n_removal", 2),
            time_limit=getattr(config, "time_limit", 60.0),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
