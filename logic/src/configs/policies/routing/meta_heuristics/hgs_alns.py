"""
HGS-ALNS (Hybrid Genetic Search with ALNS Education) configuration for Hydra.

Attributes:
    HGSALNSConfig: Configuration for the HGS-ALNS policy.

Example:
    >>> from configs.policies.hgs_alns import HGSALNSConfig
    >>> config = HGSALNSConfig()
    >>> config.hgs.n_removal
    10
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.configs.policies.routing.meta_heuristics.hgs import HGSConfig


@dataclass
class HGSALNSConfig:
    """
    Configuration for Hybrid Genetic Search with Adaptive Large Neighborhood Search Education (HGS-ALNS) policy.

    Uses ALNS for education phase and HGS for routing phase, combining the strengths
    of both metaheuristics.
    Attributes:
        None
    """

    # HGS-ALNS specific parameters
    profit_aware_operators: bool = False

    # Nested component configs
    hgs: HGSConfig = field(default_factory=HGSConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # Common policy fields
    packing: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
    time_limit: float = 60.0
    seed: Optional[int] = None

    @classmethod
    def from_config(cls, config: Any) -> "HGSALNSConfig":
        """Load the canonical nested component configs."""
        from collections.abc import Mapping
        from dataclasses import fields

        read = config.get if isinstance(config, Mapping) else lambda key, default: getattr(config, key, default)
        hgs = read("hgs", None)
        alns = read("alns", None)
        if isinstance(hgs, Mapping):
            hgs = HGSConfig(**hgs)
        if isinstance(alns, Mapping):
            alns = ALNSConfig(**alns)
        values = {
            item.name: read(item.name, item.default)
            for item in fields(cls)
            if item.name not in {"hgs", "alns", "packing", "route_improvement"}
        }
        return cls(
            hgs=hgs or HGSConfig(),
            alns=alns or ALNSConfig(),
            packing=read("packing", []),
            route_improvement=read("route_improvement", []),
            **values,
        )
