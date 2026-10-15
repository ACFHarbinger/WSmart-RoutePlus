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
from typing import List, Optional

from .other.packing import PackingConfig
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
