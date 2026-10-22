"""
Progressive Hedging (PH) policy configuration.

Attributes:
    PHConfig: Configuration for the Progressive Hedging (PH) policy.

Example:
    >>> from configs.policies.ph import PHConfig
    >>> config = PHConfig()
    >>> config.rho
    1.0
    >>> config.max_iterations
    50
    >>> config.convergence_tol
    0.01
    >>> config.sub_solver
    'bc'
    >>> config.num_scenarios
    10
    >>> config.horizon
    7
    >>> config.consensus_scope
    'day_0'
    >>> config.stockout_penalty
    500.0
    >>> config.time_limit
    300.0
    >>> config.verbose
    True
    >>> config.packing
    None
    >>> config.route_improvement
    None
    >>> config.seed
    None
"""

from dataclasses import dataclass
from typing import List, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig

from logic.src.configs.policies.route_improvement import RouteImprovingConfig


@dataclass
class PHConfig:
    """Configuration for Progressive Hedging (PH) policy.

    Progressive Hedging (Rockafellar and Wets, 1991) is a horizontal
    decomposition algorithm for stochastic programming. It decomposes the
    stochastic VRP into scenario-specific subproblems and iteratively enforces
    non-anticipativity constraints.

    Attributes:
        rho: Penalty parameter for the quadratic non-anticipativity term.
            Larger values enforce consensus faster but may lead to suboptimality.
        max_iterations: Maximum number of PH iterations.
        convergence_tol: Tolerance for consensual visit decisions (non-anticipativity error).
        sub_solver: Key of the deterministic solver to use for subproblems (e.g., 'bc', 'alns').
        num_scenarios: Number of scenarios for SAA if none are provided.
        time_limit: Total wall-clock time limit in seconds.
        verbose: Enable detailed logging of PH iterations and convergence stats.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        seed: Random seed for reproducibility.
    """

    rho: float = 1.0
    max_iterations: int = 50
    convergence_tol: float = 0.01
    sub_solver: str = "bc"
    num_scenarios: int = 10
    horizon: int = 7
    consensus_scope: str = "day_0"  # Options: "day_0", "full"
    stockout_penalty: float = 500.0
    time_limit: float = 300.0
    verbose: bool = True
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    seed: Optional[int] = None
