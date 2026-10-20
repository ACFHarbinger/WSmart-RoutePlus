"""
HGS (Hybrid Genetic Search) configuration.

Attributes:
    HGSConfig: Configuration for the HGS (Hybrid Genetic Search) policy.

Example:
    >>> from configs.policies.hgs import HGSConfig
    >>> config = HGSConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import List, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.acceptance_criteria import AcceptanceConfig, OnlyImprovingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class HGSConfig:
    """Configuration for Hybrid Genetic Search (HGS) policy.
    Based on Vidal et al. (2022) - "Hybrid genetic search for the CVRP".

    Attributes:
        restart_timer: Maximum wall-clock seconds for optimization, before restarting the algorithm (0 = unlimited)
        time_limit: Maximum time in seconds for the solver (0.0 = unlimited).
        seed: Random seed for reproducibility.
        mu: Minimum population size per subpopulation.
        n_offspring: Generation size (number before survivor selection).
        nb_elite: Number of elite individuals to preserve.
        nb_close: Number of close individuals for diversity measurement.
        target_feasible: Target proportion of feasible solutions.
        n_iterations_no_improvement: Max iterations without improvement.
        repair_probability: Probability of repairing infeasible offspring.
        crossover_rate: Probability of applying crossover.
        max_vehicles: Maximum number of vehicles (0 for unlimited).
        initial_penalty_capacity: Initial penalty for capacity violations.
        penalty_increase: Multiplier for increasing penalty.
        penalty_decrease: Multiplier for decreasing penalty.
        local_search_iterations: Number of local search iterations.
        engine: Solver engine to use ('custom', 'pyvrp').
        profit_aware_operators: Whether to use profit-aware operators.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        acceptance_criterion: Acceptance criterion config for local search.
    """

    # Core HGS parameters (Vidal 2022)
    time_limit: float = 60.0
    restart_timer: float = 60.0
    seed: Optional[int] = None
    mu: int = 25
    n_offspring: int = 40
    nb_elite: int = 4
    nb_close: int = 5
    target_feasible: float = 0.2
    n_iterations_no_improvement: int = 20000

    # Genetic operators
    repair_probability: float = 0.5
    crossover_rate: float = 1.0

    # Vehicle constraints
    max_vehicles: int = 0

    # Penalty management
    initial_penalty_capacity: float = 1.0
    penalty_increase: float = 1.2
    penalty_decrease: float = 0.85

    # Local search
    local_search_iterations: int = 500
    nb_granular: int = 20  # Used by the HGS-ALNS composite.
    mutation_rate: float = 1.0  # Used by the HGS-ALNS composite.

    # Engine configuration
    engine: str = "custom"
    profit_aware_operators: bool = True
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    acceptance_criterion: AcceptanceConfig = field(
        default_factory=lambda: AcceptanceConfig(method="oi", params=OnlyImprovingConfig())
    )
    lambda_param: Optional[int] = None

    def __post_init__(self) -> None:
        if self.lambda_param is not None and self.n_offspring == 40:
            self.n_offspring = self.lambda_param
        elif self.lambda_param is None:
            self.lambda_param = self.n_offspring
