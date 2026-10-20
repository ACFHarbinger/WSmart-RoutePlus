"""
SISR (Slack Induction by String Removal) configuration.

Attributes:
    SISRConfig: Configuration for the Slack Induction by String Removal (SISR) policy.

Example:
    >>> from configs.policies.sisr import SISRConfig
    >>> config = SISRConfig()
    >>> config.time_limit
    10.0
    >>> config.max_iterations
    1000
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from logic.src.configs.policies.other.packing_dispatch import PackingConfig

from .other.route_improvement import RouteImprovingConfig


@dataclass
class SISRConfig:
    """Configuration for Slack Induction by String Removal (SISR) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        max_iterations: Maximum number of SISR iterations.
        start_temp: Initial temperature for simulated annealing acceptance.
        cooling_rate: Rate at which temperature decreases each iteration.
        max_string_len: Maximum length of string to remove.
        avg_string_len: Average length of strings to remove.
        blink_rate: Probability of 'blinking' during insertion.
        destroy_ratio: Fraction of solution to destroy per iteration.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    time_limit: float = 10.0
    seed: Optional[int] = None
    max_iterations: int = 1000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    max_string_len: int = 10
    avg_string_len: float = 3.0
    blink_rate: float = 0.01
    destroy_ratio: float = 0.2
    profit_aware_operators: bool = False
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None

    @classmethod
    def from_config(cls, config: Any) -> "SISRConfig":
        """Create SISRConfig from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            "SISRConfig" instance with values from config.
        """
        return cls(
            time_limit=getattr(config, "time_limit", 10.0),
            max_iterations=getattr(config, "max_iterations", 1000),
            start_temp=getattr(config, "start_temp", 100.0),
            cooling_rate=getattr(config, "cooling_rate", 0.995),
            max_string_len=getattr(config, "max_string_len", 10),
            avg_string_len=getattr(config, "avg_string_len", 3.0),
            blink_rate=getattr(config, "blink_rate", 0.01),
            destroy_ratio=getattr(config, "destroy_ratio", 0.2),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
        )
