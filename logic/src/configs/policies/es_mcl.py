"""
(μ,λ) Evolution Strategy configuration.

Strictly follows the generational evolutionary algorithm terminology where:
- μ (mu): Parent population size.
- λ (lambda): Offspring population size.

Attributes:
    MuCommaLambdaESConfig: Attributes for ES-MCL configuration.

Example:
    >>> from configs.policies.es_mcl import MuCommaLambdaESConfig
    >>> config = MuCommaLambdaESConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.route_improvement import RouteImprovingConfig


@dataclass
class MuCommaLambdaESConfig:
    """Configuration for (μ,λ) Evolution Strategy policy.

    The (μ,λ) scheme is a non-elitist generational strategy where the offspring
    entirely replace the parents.

    Attributes:
        mu (int): Number of parents.
        lambda_ (int): Number of offspring.
        n_removal (int): Number of routes to remove.
        max_iterations (int): Maximum number of iterations.
        local_search_iterations (int): Number of local search iterations.
        time_limit (float): Time limit in seconds.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        seed (Optional[int]): Seed for the random number generator.
        packing (Optional[List[PackingConfig]]): Packing configurations.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement configurations.
    """

    mu: int = 15
    lambda_: int = 100
    n_removal: int = 3
    max_iterations: int = 500
    local_search_iterations: int = 100
    time_limit: float = 60.0
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    @classmethod
    def from_config(cls, config: Any) -> "MuCommaLambdaESConfig":
        """Build parameters from a configuration object.

        Args:
            config: Configuration object or dictionary.

        Returns:
            "MuCommaLambdaESConfig" instance.
        """
        import dataclasses

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            mu=getattr(config, "mu", 15),
            lambda_=getattr(config, "lambda_", 100),
            n_removal=getattr(config, "n_removal", 3),
            max_iterations=getattr(config, "max_iterations", 500),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
