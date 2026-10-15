"""
SCA (Sine Cosine Algorithm) configuration for Hydra.

Attributes:
    SCAConfig: Configuration for the Sine Cosine Algorithm (SCA) policy.

Example:
    >>> from configs.policies.sca import SCAConfig
    >>> config = SCAConfig()
    >>> config.pop_size
    20
    >>> config.max_iterations
    200
    >>> config.time_limit
    60.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class SCAConfig:
    """
    Configuration for the Sine Cosine Algorithm policy.

    Attributes:
        pop_size: Population size.
        a_max: Initial control parameter value (linearly decays to 0).
        max_iterations: Maximum SCA iterations.
        time_limit: Wall-clock time limit in seconds.
        packing: Packing algorithm config list.
        route_improvement: Route improvement operation config list.
    """

    pop_size: int = 20
    a_max: float = 2.0
    max_iterations: int = 200
    time_limit: float = 60.0
    seed: Optional[int] = None
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    local_search_iterations: int = 500
    profit_aware_operators: bool = False


    @classmethod
    def from_config(cls, config: Any) -> "SCAConfig":
        """Create SCAConfig from a configuration object.

        Args:
            config: Configuration object or dictionary.

        Returns:
            "SCAConfig" instance with values from config.
        """
        import dataclasses
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in dataclasses.fields(cls)}})

        return cls(
            pop_size=getattr(config, "pop_size", 20),
            a_max=getattr(config, "a_max", 2.0),
            max_iterations=getattr(config, "max_iterations", 200),
            local_search_iterations=getattr(config, "local_search_iterations", 500),
            time_limit=getattr(config, "time_limit", 60.0),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            seed=getattr(config, "seed", None),
        )
