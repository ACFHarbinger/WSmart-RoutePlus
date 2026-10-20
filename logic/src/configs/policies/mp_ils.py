"""
Configuration for the Multi-Period Iterated Local Search (MP-ILS).

Attributes:
    MP_ILS_Config: Configuration for the Multi-Period Iterated Local Search (MP-ILS) policy.

Example:
    >>> from configs.policies.mp_ils import MP_ILS_Config
    >>> config = MP_ILS_Config()
    >>> config.iters
    50
    >>> config.perturb_size
    3
    >>> config.seed
    42
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.route_improvement import RouteImprovingConfig


@dataclass
class MP_ILS_Config:
    """
    Configuration for the Multi-Period Iterated Local Search (MP-ILS) policy.

    Attributes:
        iters (int): Number of iterations. Defaults to 50.
        perturb_size (int): Number of nodes to remove during perturbation. Defaults to 3.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 50
    perturb_size: int = 3
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None

    @classmethod
    def from_config(cls, config: Any) -> "MP_ILS_Config":
        """Create MP_ILS_Config from a configuration object or dictionary.

        Args:
            config: The configuration object or dictionary.

        Returns:
            "MP_ILS_Config": The instantiated parameters.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            iters=getattr(config, "iters", getattr(config, "max_iter", 50)),
            perturb_size=getattr(config, "perturb_size", 3),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MP_ILS_Config to a dictionary for backend compatibility.

        Args:
            None.

        Returns:
            Dict[str, Any]: A dictionary of parameter names and values.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}
