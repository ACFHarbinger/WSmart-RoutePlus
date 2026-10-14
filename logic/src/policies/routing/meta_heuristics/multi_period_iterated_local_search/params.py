"""
Configuration parameters for the Multi-Period Iterated Local Search (MP-ILS).

Attributes:
    MP_ILS_Params: Configuration parameters for MP-ILS.

Example:
    >>> params = MP_ILS_Params(iters=100, perturb_size=5)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class MP_ILS_Params:
    """
    Configuration parameters for MP-ILS.

    Attributes:
        iters: Number of iterations.
        perturb_size: Number of nodes to remove during perturbation.
        seed: Random seed.
    """

    iters: int = 50
    perturb_size: int = 3
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> MP_ILS_Params:
        """Create MP_ILS_Params from a configuration object or dictionary.

        Args:
            config: The configuration object or dictionary.

        Returns:
            MP_ILS_Params: The instantiated parameters.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            iters=getattr(config, "iters", getattr(config, "max_iter", 50)),
            perturb_size=getattr(config, "perturb_size", 3),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert MP_ILS_Params to a dictionary for backend compatibility.

        Args:
            None.

        Returns:
            Dict[str, Any]: A dictionary of parameter names and values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
