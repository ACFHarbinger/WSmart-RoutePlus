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
from typing import Any, List

from .alns import ALNSConfig
from .hgs import HGSConfig


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
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Nested component configs
    hgs: HGSConfig = field(default_factory=HGSConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # Common policy fields
    packing: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
