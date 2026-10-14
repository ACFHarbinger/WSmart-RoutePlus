"""
Configuration parameters for the Matheuristic Hyper-Heuristic (MHH).

Attributes:
    MHHParams: Configuration parameters for the MHH solver.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics import MHHParams
    >>> params = MHHParams()
    >>> print(params)
    MHHParams(
        iters=10,
        seed=42,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class MHHParams:
    """
    Configuration parameters for MHH.

    Attributes:
        iters: Number of iterations.
        seed: Random seed.
    """

    iters: int = 10
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> MHHParams:
        """
        Create MHHParams from a configuration object or dictionary.

        Args:
            config: Configuration object or dictionary.

        Returns:
            MHHParams: Parameters for the MHH solver.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            iters=getattr(config, "iters", 10),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert MHHParams to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Dictionary representation of MHHParams.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
