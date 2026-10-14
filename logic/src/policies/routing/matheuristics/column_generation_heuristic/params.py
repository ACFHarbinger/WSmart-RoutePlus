"""
Configuration parameters for the Column Generation Heuristic (CGH).

Attributes:
    CGHParams: Dataclass holding CGH solver hyperparameters.

Example:
    >>> from logic.src.policies.routing.matheuristics.column_generation_heuristic.params import CGHParams
    >>> params = CGHParams(cg_iters=10, routes_per_iter=50)
    >>> d = params.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class CGHParams:
    """
    Configuration parameters for Column Generation Heuristic.

    Attributes:
        cg_iters: Number of column generation iterations.
        routes_per_iter: Number of heuristic routes to generate per iteration.
        seed: Random seed.
    """

    cg_iters: int = 10
    routes_per_iter: int = 50
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> CGHParams:
        """Create CGHParams from a configuration object or dictionary.

        Args:
            config: Configuration object or dict with CGH parameter attributes.

        Returns:
            CGHParams: Populated parameter dataclass.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            cg_iters=getattr(config, "cg_iters", 10),
            routes_per_iter=getattr(config, "routes_per_iter", 50),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert CGHParams to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Mapping of parameter names to their values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
