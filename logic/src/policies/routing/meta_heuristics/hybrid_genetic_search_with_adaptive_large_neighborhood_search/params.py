"""
Parameters for the Hybrid Genetic Search with Adaptive Large Neighborhood Search (HGS-ALNS).

Attributes:
    HGSALNSParams: Configuration parameters for the HGS-ALNS solver.

Example:
    >>> params = HGSALNSParams()
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from logic.src.configs.policies.alns import ALNSConfig
from logic.src.configs.policies.hgs import HGSConfig


@dataclass
class HGSALNSParams:
    """
    Parameters for the Hybrid Genetic Search with ALNS Education metaheuristic.

    Uses ALNS for the education phase and HGS for the routing phase, combining
    the exploration power of genetic algorithms with the exploitation strength
    of adaptive large neighborhood search.

    Attributes:
        time_limit: Maximum execution time in seconds.
        seed: Random seed for reproducibility.
        vrpp: Whether to solve for VRP with Profits.
        profit_aware_operators: Whether to use profit-aware neighborhood moves.
        hgs_params: Parameters for the HGS component.
        alns_params: Parameters for the ALNS component.
    """

    # Hybrid-specific parameters
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False

    # HGS Components (Genetic Evolution & Population Management)
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

    # ALNS Components (Education & Intensive Local Search)
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
    def from_config(cls, config: Any) -> "HGSALNSParams":
        """Create HGSALNSParams from a configuration object.

        Args:
            config: Configuration source for parameters.

        Returns:
            HGSALNSParams: The initialized parameters object.
        """
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
            vrpp=getattr(config, "vrpp", True),
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
            "vrpp": self.vrpp,
            "profit_aware_operators": self.profit_aware_operators,
        }
