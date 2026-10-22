"""
Configuration for the Multi-Period Simulated Annealing (MP-SA).

Attributes:
    MP_SA_Config: Configuration for the Multi-Period Simulated Annealing (MP-SA) policy.

Example:
    >>> from configs.policies.mp_sa import MP_SA_Config
    >>> config = MP_SA_Config()
    >>> config.iters
    500
    >>> config.init_temp
    100.0
    >>> config.cooling_rate
    0.95
    >>> config.seed
    42
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig

from logic.src.configs.policies.route_improvement import RouteImprovingConfig


@dataclass
class MP_SA_Config:
    """
    Configuration for the Multi-Period Simulated Annealing (MP-SA) policy.

    Attributes:
        iters (int): Number of iterations. Defaults to 500.
        init_temp (float): Initial temperature. Defaults to 100.0.
        cooling_rate (float): Cooling rate. Defaults to 0.95.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 500
    init_temp: float = 100.0
    cooling_rate: float = 0.95
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None

    @classmethod
    def from_config(cls, config: Any) -> "MP_SA_Config":
        """Create MP_SA_Config from a configuration object or dictionary.

        Args:
            config: The configuration object or dictionary.

        Returns:
            "MP_SA_Config": The instantiated parameters.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            iters=getattr(config, "iters", getattr(config, "max_iter", 500)),
            init_temp=getattr(config, "init_temp", 100.0),
            cooling_rate=getattr(config, "cooling_rate", 0.95),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MP_SA_Config to a dictionary for backend compatibility.

        Args:
            None.

        Returns:
            Dict[str, Any]: A dictionary of parameter names and values.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}
