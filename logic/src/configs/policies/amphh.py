"""
Configuration for the Adaptive Memory Programming Hyper-Heuristic (AMP-HH).

Attributes:
    AMPHHConfig: Attributes for AMP-HH configuration.

Example:
    >>> from configs.policies.amphh import AMPHHConfig
    >>> config = AMPHHConfig()
    >>> config.mem_size
    10
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class AMPHHConfig:
    """
    Configuration for the Adaptive Memory Programming Hyper-Heuristic (AMP-HH) policy.

    AMP-HH maintains a memory of high-quality components and reconstructs
    new solutions by combining these components using Low-Level Heuristics (LLHs).

    Attributes:
        mem_size (int): Size of the adaptive memory. Defaults to 10.
        iters (int): Number of iterations. Defaults to 50.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    mem_size: int = 10
    iters: int = 50
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None


    @classmethod
    def from_config(cls, config: Any) -> "AMPHHConfig":
        """Create AMPHHConfig from a configuration object or dictionary.

        Args:
            config (Any): Configuration for the policy.

        Returns:
            "AMPHHConfig": Configuration parameters for AMPHH.
        """
        from dataclasses import fields
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            mem_size=getattr(config, "mem_size", 10),
            iters=getattr(config, "iters", 50),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert AMPParams to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Dictionary representation of AMPHHConfig.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)}
