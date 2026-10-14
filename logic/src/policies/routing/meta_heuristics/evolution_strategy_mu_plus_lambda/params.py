r"""Parameters for the (μ+λ) Evolution Strategy.

Attributes:
    MuPlusLambdaESParams: Parameter dataclass for (μ+λ)-ES.

Example:
    >>> params = MuPlusLambdaESParams(mu=10, lambda_=5)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MuPlusLambdaESParams:
    r"""
    Parameters for the (μ+λ) Evolution Strategy.

    A (μ+λ)-ES maintains a population of μ parent solutions. In each iteration,
    it generates λ offspring through recombination and mutation.
    The selection operator then selects the absolute best μ individuals from the
    combined pool of μ parents and λ offspring to form the next generation.

    Attributes:
        mu: The number of parent individuals maintained in the population (μ).
        lambda_: The number of offspring generated per generation (λ).
        n_removal: The mutation strength parameter.
        max_iterations: The generational loop limit.
        local_search_iterations: Intensity of local optimization.
        time_limit: Wall-clock duration in seconds.
        seed: Random seed for reproducibility.
        vrpp: Whether the problem is VRP with Profits.
        profit_aware_operators: Whether to use profit-aware insertion/removal.
    """

    mu: int = 10  # Parent population size (μ)
    lambda_: int = 5  # Offspring population size (λ)
    n_removal: int = 3  # Mutation strength
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    @classmethod
    def from_config(cls, config: Any) -> MuPlusLambdaESParams:
        """Build parameters from a configuration object.

        Args:
            config: Configuration source (dict or Hydra config).

        Returns:
            MuPlusLambdaESParams instance.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            mu=getattr(config, "mu", 10),
            lambda_=getattr(config, "lambda_", 5),
            n_removal=getattr(config, "n_removal", 3),
            max_iterations=getattr(config, "max_iterations", 500),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            vrpp=getattr(config, "vrpp", True),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
