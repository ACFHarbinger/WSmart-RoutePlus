"""
HGS-ALNS (Hybrid Genetic Search with ALNS Education) configuration for Hydra.

Attributes:
    HGSALNSConfig: Configuration for the HGS-ALNS policy.

Example:
    >>> from configs.policies.hgs_alns import HGSALNSConfig
    >>> config = HGSALNSConfig()
    >>> config.hgs.n_removal
    10
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.hgs import HGSConfig


@dataclass
class HGSALNSConfig:
    """
    Configuration for Hybrid Genetic Search with Adaptive Large Neighborhood Search Education (HGS-ALNS) policy.

    Uses ALNS for education phase and HGS for routing phase, combining the strengths
    of both metaheuristics.
    Attributes:
        None
    """

    # HGS-ALNS specific parameters
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Nested component configs
    hgs: HGSConfig = field(default_factory=HGSConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # Common policy fields
    packing: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
    time_limit: float = 60.0
    seed: Optional[int] = None
    hgs_params: HGSConfig = field(
        default_factory=lambda: HGSConfig(
            time_limit=60.0,
            mu=50,
            nb_elite=10,
            mutation_rate=0.2,
            crossover_rate=0.7,
            lambda_param=40,  # Default for lambda_param
            n_iterations_no_improvement=10,
            nb_granular=10,
            local_search_iterations=500,
            max_vehicles=0,
        )
    )
    alns_params: ALNSConfig = field(
        default_factory=lambda: ALNSConfig(
            time_limit=60.0,
            max_iterations=50,
            start_temp=100.0,
            cooling_rate=0.995,
            reaction_factor=0.1,
            max_removal_pct=0.3,
        )
    )
