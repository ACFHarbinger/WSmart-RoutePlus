"""
Configuration for the Selection Hyper-Heuristic (SHH).

Attributes:
    SHHConfig: Configuration for the Selection Hyper-Heuristic (SHH) policy.

Example:
    >>> from configs.policies.shh import SHHConfig
    >>> config = SHHConfig()
    >>> config.iters
    200
    >>> config.history_len
    10
    >>> config.packing
    None
    >>> config.route_improvement
    None
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.route_improvement import RouteImprovingConfig


@dataclass
class SHHConfig:
    """
    Configuration for the Selection Hyper-Heuristic (SHH) policy.

    SHH adaptively selects a Low-Level Heuristic (LLH) based on past
    performance and applies it to the current solution. It uses Late
    Acceptance Hill Climbing (LAHC) for acceptance.

    Attributes:
        iters (int): Number of iterations. Defaults to 200.
        history_len (int): Length of the Late Acceptance history. Defaults to 10.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 200
    history_len: int = 10
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None

    @classmethod
    def from_config(cls, config: Any) -> "SHHConfig":
        """
        Create SHHConfig from a configuration object or dictionary.

        Args:
            config: Configuration object or dictionary.

        Returns:
            "SHHConfig": Parameters for the SHH solver.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            iters=getattr(config, "iters", 200),
            history_len=getattr(config, "history_len", 10),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert SHHConfig to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Dictionary representation of SHHConfig.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}
