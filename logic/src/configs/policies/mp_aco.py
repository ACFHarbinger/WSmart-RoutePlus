"""
Configuration for the Multi-Period Ant Colony Optimization (MP-ACO).

Attributes:
    MP_ACO_Config: Configuration for the Multi-Period Ant Colony Optimization (MP-ACO) policy.

Example:
    >>> from configs.policies.mp_aco import MP_ACO_Config
    >>> config = MP_ACO_Config()
    >>> config.iters
    50
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.route_improvement import RouteImprovingConfig


@dataclass
class MP_ACO_Config:
    """
    Configuration for the Multi-Period Ant Colony Optimization (MP-ACO) policy.

    Attributes:
        n_ants (int): Number of ants in the colony. Defaults to 10.
        iters (int): Number of iterations. Defaults to 50.
        alpha (float): Influence of pheromone on path selection. Defaults to 1.0.
        beta (float): Influence of heuristic information on path selection. Defaults to 2.0.
        rho (float): Pheromone evaporation rate. Defaults to 0.1.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    n_ants: int = 10
    iters: int = 50
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None

    @classmethod
    def from_config(cls, config: Any) -> "MP_ACO_Config":
        """Create MP_ACO_Config from a configuration object or dictionary.

        Args:
            config: The configuration object or dictionary.

        Returns:
            "MP_ACO_Config": The instantiated parameters.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            n_ants=getattr(config, "n_ants", 10),
            iters=getattr(config, "iters", 50),
            alpha=getattr(config, "alpha", 1.0),
            beta=getattr(config, "beta", 2.0),
            rho=getattr(config, "rho", 0.1),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MP_ACO_Config to a dictionary for backend compatibility.

        Args:
            None.

        Returns:
            Dict[str, Any]: A dictionary of parameter names and values.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}
