"""
PG-CLNS (Pheromone-Guided Cooperative Large Neighborhood Search) configuration.

Attributes:
    PGCLNSConfig: Attributes for PG-CLNS configuration.

Example:
    >>> from configs.policies.pg_clns import PGCLNSConfig
    >>> config = PGCLNSConfig()
    >>> config.n_teams
    10
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig


@dataclass
class PGCLNSLNSConfig:
    """Coaching LNS configuration, preserving PG-CLNS's shared RNG stream."""

    max_iterations: int = 100
    time_limit: float = 30.0
    seed: Optional[int] = None
    reaction_factor: float = 0.5
    xi: float = 0.2
    acceptance_criterion: AcceptanceConfig = field(
        default_factory=lambda: AcceptanceConfig(
            method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.95)
        )
    )


@dataclass
class PGCLNSConfig:
    """Configuration for Pheromone-Guided Cooperative Large Neighborhood Search (PG-CLNS) policy.

    Attributes:
        population_size: Population size.
        max_iterations: Number of cooperative search iterations.
        replacement_rate: Fraction of population replaced by pheromone-guided construction.
        time_limit: Overall time limit in seconds.
        seed: Random seed for reproducibility.
        aco: Nested configuration for the ACO component.
        lns: Nested configuration for the LNS component.
        profit_aware_operators: Whether to use operators aware of node profits.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    # Population Parameters
    population_size: int = 10
    max_iterations: int = 50
    replacement_rate: float = 0.2

    # Global Parameters
    time_limit: float = 60.0
    seed: Optional[int] = None

    # Nested component configs
    aco: KSparseACOConfig = field(
        default_factory=lambda: KSparseACOConfig(
            n_ants=10,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            tau_0=1.0,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,  # Only one iteration per construction phase
            time_limit=30.0,
            local_search=False,
            local_search_iterations=0,
            elitist_weight=1.0,
        )
    )
    lns: PGCLNSLNSConfig = field(default_factory=PGCLNSLNSConfig)

    # Common policy fields
    profit_aware_operators: bool = True
    packing: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
