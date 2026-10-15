"""
Configuration for the Matheuristic Hyper-Heuristic (MHH).

Attributes:
    MHHConfig: Configuration for the Matheuristic Hyper-Heuristic (MHH) policy.

Example:
    >>> from configs.policies.mhh import MHHConfig
    >>> config = MHHConfig()
    >>> config.iters
    10
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MHHConfig:
    """
    Configuration for the Matheuristic Hyper-Heuristic (MHH) policy.

    MHH applies matheuristic Low-Level Heuristics (LLHs) iteratively to
    find improvements in the multi-period plan.

    Attributes:
        iters (int): Number of iterations. Defaults to 10.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 10
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None


    @classmethod
    def from_config(cls, config: Any) -> "MHHConfig":
        """
        Create MHHConfig from a configuration object or dictionary.

        Args:
            config: Configuration object or dictionary.

        Returns:
            "MHHConfig": Parameters for the MHH solver.
        """
        from dataclasses import fields
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            iters=getattr(config, "iters", 10),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert MHHConfig to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Dictionary representation of MHHConfig.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)}
