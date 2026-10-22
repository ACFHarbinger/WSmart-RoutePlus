"""
Configuration for the Lagrangian Relaxation Heuristic (LRH).

Attributes:
    LRHConfig: Configuration for the Lagrangian Relaxation Heuristic (LRH) policy.

Example:
    >>> from configs.policies.lrh import LRHConfig
    >>> config = LRHConfig()
    >>> config.max_iter
    50
    >>> config.step_size
    2.0
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig

from logic.src.configs.policies.route_improvement import RouteImprovingConfig


@dataclass
class LRHConfig:
    """
    Configuration for the Lagrangian Relaxation Heuristic (LRH) policy.

    LRH relaxes the capacity constraints using Lagrangian multipliers and
    solves a simplified problem in each iteration. It uses a subgradient
    method to update the multipliers and find dual bounds.

    Attributes:
        max_iter (int): Maximum number of subgradient iterations. Defaults to 50.
        step_size (float): Initial step size for subgradient method. Defaults to 2.0.
        halving_freq (int): Frequency (iterations) at which to halve the step size.
            Defaults to 10.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    max_iter: int = 50
    step_size: float = 2.0
    halving_freq: int = 10
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None

    @classmethod
    def from_config(cls, config: Any) -> "LRHConfig":
        """Create LRHConfig from a configuration object or dictionary.

        Args:
            config: Configuration object or dict with LRH parameter attributes.

        Returns:
            "LRHConfig": Populated parameter dataclass.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            max_iter=getattr(config, "max_iter", 50),
            step_size=getattr(config, "step_size", 2.0),
            halving_freq=getattr(config, "halving_freq", 10),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert LRHConfig to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Mapping of parameter names to their values.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}
