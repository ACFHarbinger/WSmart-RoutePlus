"""
(μ,κ,λ) Evolution Strategy configuration with age-based selection.

Parents exceeding age κ are excluded from selection, preventing stagnation.

Attributes:
    MuKappaLambdaESConfig: Configuration for (μ,κ,λ) Evolution Strategy policy.

Example:
    >>> from configs.policies.es_mkl import MuKappaLambdaESConfig
    >>> config = MuKappaLambdaESConfig()
    >>> config.mu
    15
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.route_improvement import RouteImprovingConfig


@dataclass
class MuKappaLambdaESConfig:
    """Configuration for (μ,κ,λ) Evolution Strategy policy.

    Evolution strategy with age-limited parent survival for long-term optimization.

    Attributes:
        mu (int): Number of parents.
        kappa (int): Maximum age (generations).
        lambda_ (int): Number of offspring.
        rho (int): Recombination participants.
        tau_local (float): Local learning rate.
        tau_global (float): Global learning rate.
        initial_sigma (float): Initial mutation standard deviation.
        min_sigma (float): Minimum mutation standard deviation.
        max_sigma (float): Maximum mutation standard deviation.
        recombination_type (str): Type of recombination.
        bounds_min (Optional[float]): Minimum bound for decision variables.
        bounds_max (Optional[float]): Maximum bound for decision variables.
        n_removal (int): Number of routes to remove.
        stagnation_limit (int): Stagnation limit.
        local_search_iterations (int): Number of local search iterations.
        max_iterations (int): Maximum number of iterations.
        time_limit (float): Time limit in seconds.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        seed (Optional[int]): Seed for the random number generator.
        packing (Optional[List[PackingConfig]]): Packing configurations.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement configurations.
    """

    # Population parameters (μ,κ,λ)
    mu: int = 15  # Number of parents
    kappa: int = 7  # Maximum age (generations)
    lambda_: int = 100  # Number of offspring
    rho: int = 2  # Recombination participants

    # Learning rate
    tau_local: float = 1.0 / (2.0**0.5)  # τ_local = 1/√(2n) for n-dimensional
    tau_global: float = 1.0 / (2.0 * (2.0**0.5))  # τ_global = 1/(2√n)

    # Recombination
    initial_sigma: float = 1.0
    min_sigma: float = 1e-10
    max_sigma: float = 10.0
    recombination_type: str = "intermediate"  # 'intermediate' or 'discrete'

    # Decision variables
    bounds_min: Optional[float] = -5.0
    bounds_max: Optional[float] = 5.0

    # Mutation parameters (routing-specific)
    n_removal: int = 3  # Nodes removed in destroy-repair

    # Restart mechanism
    stagnation_limit: int = 10  # Unused in (μ,κ,λ)-ES (age control replaces this)

    # Local search
    local_search_iterations: int = 100

    # Runtime control
    max_iterations: int = 500
    time_limit: float = 60.0

    # Infrastructure
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    def __post_init__(self):
        """Validate parameters.

        Returns:
            None.
        """
        assert self.mu > 0, "μ must be positive"
        assert self.kappa > 0, "κ must be positive"
        assert self.lambda_ >= self.mu, "λ must be >= μ for (μ,κ,λ)-selection"
        assert self.rho >= 1 and self.rho <= self.mu, "ρ must be in [1, μ]"
        assert self.recombination_type in ["intermediate", "discrete"], (
            "recombination_type must be 'intermediate' or 'discrete'"
        )
        assert self.initial_sigma > 0, "initial_sigma must be positive"
        assert self.min_sigma > 0 and self.min_sigma < self.max_sigma, (
            "min_sigma must be positive and less than max_sigma"
        )

    @classmethod
    def from_config(cls, config: Any) -> "MuKappaLambdaESConfig":
        """Build parameters from a configuration object.

        Args:
            config: Configuration object or dictionary.

        Returns:
            "MuKappaLambdaESConfig" instance.
        """
        import dataclasses

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            mu=getattr(config, "mu", 15),
            kappa=getattr(config, "kappa", 7),
            lambda_=getattr(config, "lambda_", 100),
            rho=getattr(config, "rho", 2),
            tau_local=getattr(config, "tau_local", 1.0 / (2.0**0.5)),
            tau_global=getattr(config, "tau_global", 1.0 / (2.0 * (2.0**0.5))),
            initial_sigma=getattr(config, "initial_sigma", 1.0),
            recombination_type=getattr(config, "recombination_type", "intermediate"),
            max_iterations=getattr(config, "max_iterations", 1000),
            time_limit=getattr(config, "time_limit", 300.0),
            min_sigma=getattr(config, "min_sigma", 1e-10),
            max_sigma=getattr(config, "max_sigma", 10.0),
            bounds_min=getattr(config, "bounds_min", -5.0),
            bounds_max=getattr(config, "bounds_max", 5.0),
            n_removal=getattr(config, "n_removal", 3),
            stagnation_limit=getattr(config, "stagnation_limit", 10),
            local_search_iterations=getattr(config, "local_search_iterations", 100),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
