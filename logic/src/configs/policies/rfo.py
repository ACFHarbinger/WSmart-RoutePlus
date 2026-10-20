"""
Configuration for the Relax-and-Fix-and-Optimize (RFO) matheuristic.

Attributes:
    RFOConfig: Configuration for the Relax-and-Fix-and-Optimize (RFO) policy.

Example:
    >>> from configs.policies.rfo import RFOConfig
    >>> config = RFOConfig()
    >>> config.window_size
    3
    >>> config.step_size
    2
    >>> config.mip_time
    60.0
    >>> config.mip_gap
    0.01
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
class RFOConfig:
    """
    Configuration for the Relax-and-Fix-and-Optimize (RFO) policy.

    RFO decomposes the planning horizon into overlapping windows, solving
    successive sub-MIPs where early periods are fixed or solved as integer,
    and later periods are relaxed.

    Attributes:
        window_size (int): Size of the rolling integer window. Defaults to 3.
        step_size (int): Number of days to slide the window forward after each solve. Defaults to 2.
        mip_time (float): Time limit for each sub-MIP solve in seconds. Defaults to 60.0.
        mip_gap (float): Target optimality gap for sub-MIP solves. Defaults to 0.01.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    window_size: int = 3
    step_size: int = 2
    mip_time: float = 60.0
    mip_gap: float = 0.01

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None

    @classmethod
    def from_config(cls, config: Any) -> "RFOConfig":
        """Create RFOConfig from a configuration object or dictionary.

        Args:
            config: Configuration object.

        Returns:
            "RFOConfig": RFO parameters.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            window_size=getattr(config, "window_size", 3),
            step_size=getattr(config, "step_size", 2),
            mip_time=getattr(config, "mip_time", 60.0),
            mip_gap=getattr(config, "mip_gap", 0.01),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RFOConfig to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: RFO parameters as dictionary.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}
