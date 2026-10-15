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
from typing import Any, Dict, List, Optional

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


    @classmethod
    def from_config(cls, config: Any) -> "HGSALNSConfig":
        """Create HGSALNSConfig from a configuration object.

        Args:
            config: Configuration source for parameters.

        Returns:
            HGSALNSConfig: The initialized parameters object.
        """
        import dataclasses
        hgs_source = getattr(config, "hgs", config)
        alns_source = getattr(config, "alns", config)
        if isinstance(hgs_source, HGSConfig):
            hgs_params = hgs_source
        elif isinstance(hgs_source, dict):
            valid_keys = {f.name for f in dataclasses.fields(HGSConfig)}
            hgs_params = HGSConfig(**{k: v for k, v in hgs_source.items() if k in valid_keys})
        else:
            hgs_params = HGSConfig()

        if isinstance(alns_source, ALNSConfig):
            alns_params = alns_source
        elif isinstance(alns_source, dict):
            valid_keys = {f.name for f in dataclasses.fields(ALNSConfig)}
            alns_params = ALNSConfig(**{k: v for k, v in alns_source.items() if k in valid_keys})
        else:
            alns_params = ALNSConfig()

        return cls(
            time_limit=getattr(config, "time_limit", 60.0),
            seed=getattr(config, "seed", None),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            hgs_params=hgs_params,
            alns_params=alns_params,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary.

        Args:
            None.

        Returns:
            Dict[str, Any]: Dictionary representation of parameters.
        """
        return {
            "time_limit": self.time_limit,
            "seed": self.seed,
            "profit_aware_operators": self.profit_aware_operators,
        }
