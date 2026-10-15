"""
Configuration for the Population Hyper-Heuristic (PHH).

Attributes:
    PHHConfig: Configuration for the Population Hyper-Heuristic (PHH) policy.

Example:
    >>> from configs.policies.phh import PHHConfig
    >>> config = PHHConfig()
    >>> config.pop_size
    10
    >>> config.gens
    20
    >>> config.seed
    42
    >>> config.packing
    None
    >>> config.route_improvement
    None
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class PHHConfig:
    """
    Configuration for the Population Hyper-Heuristic (PHH) policy.

    PHH maintains a population of multi-period plans and evolves them
    using Low-Level Heuristics (LLHs) and genetic operators.

    Attributes:
        pop_size (int): Size of the population. Defaults to 10.
        gens (int): Number of generations. Defaults to 20.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    pop_size: int = 10
    gens: int = 20
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None


    @classmethod
    def from_config(cls, config: Any) -> "PHHConfig":
        """
        Create PHHConfig from a configuration object or dictionary.

        Args:
            config: Configuration object or dictionary.

        Returns:
            "PHHConfig": Parameters for the PHH solver.
        """
        from dataclasses import fields
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            pop_size=getattr(config, "pop_size", 10),
            gens=getattr(config, "gens", 20),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PHHConfig to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Dictionary representation of PHHConfig.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)}
