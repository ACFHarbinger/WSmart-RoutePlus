"""
Configuration for the Column Generation Heuristic (CGH).

Attributes:
    CGHConfig: Attributes for CGH configuration.

Example:
    >>> from configs.policies.cgh import CGHConfig
    >>> config = CGHConfig()
    >>> config.cg_iters
    10
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class CGHConfig:
    """
    Configuration for the Column Generation Heuristic (CGH) policy.

    CGH uses a master problem (set packing/partitioning) and generates routes
    heuristically. Instead of solving the exact subproblem (ESPPRC), it uses
    randomized greedy and local search to find negative reduced-cost columns.

    Attributes:
        cg_iters (int): Number of column generation iterations. Defaults to 10.
        routes_per_iter (int): Number of heuristic routes to generate in each
            column generation iteration. Defaults to 50.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    cg_iters: int = 10
    routes_per_iter: int = 50
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None


    @classmethod
    def from_config(cls, config: Any) -> "CGHConfig":
        """Create CGHConfig from a configuration object or dictionary.

        Args:
            config: Configuration object or dict with CGH parameter attributes.

        Returns:
            "CGHConfig": Populated parameter dataclass.
        """
        from dataclasses import fields
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            cg_iters=getattr(config, "cg_iters", 10),
            routes_per_iter=getattr(config, "routes_per_iter", 50),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert CGHConfig to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Mapping of parameter names to their values.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)}
