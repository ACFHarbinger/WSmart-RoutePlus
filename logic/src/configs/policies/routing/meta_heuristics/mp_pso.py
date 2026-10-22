"""
Configuration for the Multi-Period Particle Swarm Optimization (MP-PSO).

Attributes:
    MP_PSO_Config: Configuration for the Multi-Period Particle Swarm Optimization (MP-PSO) policy.

Example:
    >>> from configs.policies.mp_pso import MP_PSO_Config
    >>> config = MP_PSO_Config()
    >>> config.swarm_size
    20
    >>> config.iters
    50
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig

from logic.src.configs.policies.route_improvement import RouteImprovingConfig


@dataclass
class MP_PSO_Config:
    """
    Configuration for the Multi-Period Particle Swarm Optimization (MP-PSO) policy.

    Attributes:
        swarm_size (int): Number of particles in the swarm. Defaults to 20.
        iters (int): Number of iterations. Defaults to 50.
        w (float): Inertia weight. Defaults to 0.8.
        c1 (float): Cognitive coefficient. Defaults to 2.0.
        c2 (float): Social coefficient. Defaults to 2.0.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    swarm_size: int = 20
    iters: int = 50
    w: float = 0.8
    c1: float = 2.0
    c2: float = 2.0
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None

    @classmethod
    def from_config(cls, config: Any) -> "MP_PSO_Config":
        """Create MP_PSO_Config from a configuration object or dictionary.

        Args:
            config: The configuration object or dictionary.

        Returns:
            "MP_PSO_Config": The instantiated parameters.
        """
        from dataclasses import fields

        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            swarm_size=getattr(config, "swarm_size", getattr(config, "n_particles", 20)),
            iters=getattr(config, "iters", 50),
            w=getattr(config, "w", 0.8),
            c1=getattr(config, "c1", 2.0),
            c2=getattr(config, "c2", 2.0),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MP_PSO_Config to a dictionary for backend compatibility.

        Args:
            None.

        Returns:
            Dict[str, Any]: A dictionary of parameter names and values.
        """
        from dataclasses import fields

        return {f.name: getattr(self, f.name) for f in fields(self)}
