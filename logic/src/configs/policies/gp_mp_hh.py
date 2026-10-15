"""
Configuration for the Genetic Programming Multi-Period Hyper-Heuristic (GP-MP-HH).

Attributes:
    GP_MP_HHConfig: Configuration for the GP-MP-HH Constructive Heuristic Generator policy.

Example:
    >>> from configs.policies.gp_mp_hh import GP_MP_HHConfig
    >>> config = GP_MP_HHConfig()
    >>> config.pop_size
    10
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class GP_MP_HH_Config:
    """
    Configuration for the Genetic Programming Multi-Period Hyper-Heuristic (GP-MP-HH) policy.

    GP-MP-HH evolves a sequence of Low-Level Heuristics (LLHs) using genetic
    programming to solve multi-period routing problems.

    Attributes:
        pop_size (int): Size of the GP population. Defaults to 10.
        gens (int): Number of generations. Defaults to 20.
        prog_len (int): Maximum length of a GP individual (program). Defaults to 5.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    pop_size: int = 10
    gens: int = 20
    prog_len: int = 5
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None


    @classmethod
    def from_config(cls, config: Any) -> "GP_MP_HH_Config":
        """Create GP_MP_HH_Config from a configuration object or dictionary.

        Args:
            config (Any): Configuration for the solver.

        Returns:
            "GP_MP_HH_Config": Parameters for the GP-MP-HH policy.
        """
        from dataclasses import fields
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            pop_size=getattr(config, "pop_size", 10),
            gens=getattr(config, "gens", 20),
            prog_len=getattr(config, "prog_len", 5),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert GPM_PHH_Params to a dictionary for backend compatibility.

        Returns:
            dict: Dictionary representation of the parameters.
        """
        from dataclasses import fields
        return {f.name: getattr(self, f.name) for f in fields(self)}
