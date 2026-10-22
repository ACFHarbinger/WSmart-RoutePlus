"""
HVPL Configuration for Hydra.

Reference:
    Sun, S., Ma, L., Liu, Y., & Wang, L. (2023). "Volleyball premier league
    algorithm with ACO and ALNS for simultaneous pickup–delivery location
    routing problem."

Attributes:
    HVPLConfig: Configuration for the HVPL policy.

Example:
    >>> from configs.policies.hvpl import HVPLConfig
    >>> config = HVPLConfig()
    >>> config.time_limit
    300.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.routing.hyper_heuristics.aco_hh import HyperHeuristicACOConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig

from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig


@dataclass
class HVPLConfig:
    """
    Configuration for the Hybrid Volleyball Premier League policy.

    HVPL integrates three algorithmic paradigms:
        - ACO: Intelligent population initialization (Phase 1)
        - VPL + HGS: Population evolution with genetic operators (Phase 2)
        - ALNS: Deep local search refinement (Phase 3)

    Attributes:
        n_teams (int): Number of teams for VPL.
        max_iterations (int): Maximum number of iterations.
        substitution_rate (float): Rate of substitution operator.
        crossover_rate (float): Rate of crossover operator.
        mutation_rate (float): Rate of mutation operator.
        elite_size (int): Number of elite individuals.
        aco_init_iterations (int): Number of ACO iterations for initialization.
        time_limit (float): Time limit in seconds.
        seed (Optional[int]): Seed for the random number generator.
        packing (Optional[List[PackingConfig]]): Packing configurations.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement configurations.
    """

    # VPL Population Parameters
    n_teams: int = 30
    max_iterations: int = 100
    substitution_rate: float = 0.2

    # HGS Evolution Parameters
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 3

    # Integration Parameters
    aco_init_iterations: int = 50  # ACO iterations for initialization

    # Global Parameters
    time_limit: float = 300.0
    seed: Optional[int] = None

    # Nested component configs
    aco: KSparseACOConfig = field(
        default_factory=lambda: KSparseACOConfig(
            n_ants=20,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            scale=5,
            tau_0=1.0,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,  # Only one iteration per construction
            time_limit=60.0,
            local_search=False,  # ALNS handles local search
            local_search_iterations=0,
            elitist_weight=1.0,
        )
    )
    alns: ALNSConfig = field(
        default_factory=lambda: ALNSConfig(
            max_iterations=100,
            acceptance_criterion=AcceptanceConfig(
                method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.95, start_temp_control=0.05)
            ),
            reaction_factor=0.1,
            xi=0.3,
            time_limit=60.0,
        )
    )

    # Common policy fields
    profit_aware_operators: bool = False
    packing: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
    aco_params: Optional[HyperHeuristicACOConfig] = field(default_factory=lambda: None)
    alns_params: Optional[ALNSConfig] = field(default_factory=lambda: None)

    def __post_init__(self):
        """Initialize sub-algorithm parameters with defaults if not provided.

        Args:
            None.

        Returns:
            None.
        """
        if self.aco_params is None:
            self.aco_params = HyperHeuristicACOConfig(
                n_ants=20,
                alpha=1.0,
                beta=2.0,
                rho=0.1,
                tau_0=1.0,
                eta_decay=0.5,
                max_iterations=1,  # Only one iteration per construction phase
                time_limit=60.0,
                elitism_ratio=1.0,  # ALNS handles local search
                profit_aware_operators=self.profit_aware_operators,
            )
        else:
            # Sync flags if they were already provided but maybe not consistent
            self.aco_params.profit_aware_operators = self.profit_aware_operators

        if self.alns_params is None:
            self.alns_params = ALNSConfig(
                max_iterations=100,
                acceptance_criterion=AcceptanceConfig(
                    method="bmc",
                    params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.995, start_temp_control=0.05),
                ),
                reaction_factor=0.1,
                xi=0.4,
                segment_size=10,
                time_limit=60.0,
                profit_aware_operators=self.profit_aware_operators,
            )
        else:
            self.alns_params.profit_aware_operators = self.profit_aware_operators

        # Validate constraints
        assert self.n_teams > 0, "n_teams must be positive"
        assert 0 <= self.substitution_rate <= 1, "substitution_rate must be in [0, 1]"
        assert 0 <= self.crossover_rate <= 1, "crossover_rate must be in [0, 1]"
        assert 0 <= self.mutation_rate <= 1, "mutation_rate must be in [0, 1]"
        assert self.elite_size >= 1, "elite_size must be at least 1"

    @classmethod
    def from_config(cls, config: Any) -> "HVPLConfig":
        """Create HVPLConfig from a configuration object.

        Args:
            config: Configuration source (dataclass or dictionary).

        Returns:
            HVPLConfig: Initialized parameter object.
        """
        from logic.src.policies.routing.base.base_routing_policy import _structured_load

        return cls(
            n_teams=getattr(config, "n_teams", 30),
            max_iterations=getattr(config, "max_iterations", 100),
            substitution_rate=getattr(config, "substitution_rate", 0.2),
            crossover_rate=getattr(config, "crossover_rate", 0.8),
            mutation_rate=getattr(config, "mutation_rate", 0.1),
            elite_size=getattr(config, "elite_size", 3),
            aco_init_iterations=getattr(config, "aco_init_iterations", 50),
            time_limit=getattr(config, "time_limit", 300.0),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            aco_params=_structured_load(HyperHeuristicACOConfig, getattr(config, "aco", {}))
            if getattr(config, "aco", None)
            else None,
            alns_params=_structured_load(ALNSConfig, getattr(config, "alns", {}))
            if getattr(config, "alns", None)
            else None,
        )
