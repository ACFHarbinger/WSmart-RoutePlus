"""
TSP (Traveling Salesman Problem) policy configuration.

Attributes:
    TSPConfig: Configuration for the Traveling Salesman Problem (TSP) policy.

Example:
    >>> from configs.policies.tsp import TSPConfig
    >>> config = TSPConfig()
    >>> config.cache
    True
    >>> config.time_limit
    60.0
    >>> config.packing
    None
    >>> config.route_improvement
    None
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig
from logic.src.configs.policies.route_improvement import RouteImprovingConfig


@dataclass
class TSPConfig:
    """Configuration for Traveling Salesman Problem (TSP) policy.

    Attributes:
        cache: Whether to cache solutions.
        time_limit: Maximum time in seconds for the solver.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    cache: bool = True
    engine: str = "fast_tsp"
    time_limit: float = 60.0
    seed: Optional[int] = None
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    @classmethod
    def from_config(cls, config: Any) -> "TSPConfig":
        """Create TSPConfig from a configuration object or dictionary.

        Args:
            config: Configuration object.

        Returns:
            "TSPConfig": Configuration parameters for the Traveling Salesman Problem.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            time_limit=getattr(config, "time_limit", 2.0),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert Params to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the parameters.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}
