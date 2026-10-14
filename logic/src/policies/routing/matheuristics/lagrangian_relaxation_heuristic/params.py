"""
Configuration parameters for the Lagrangian Relaxation Heuristic (LRH).

Attributes:
    LRHParams: Dataclass holding LRH solver hyperparameters.

Example:
    >>> from logic.src.policies.routing.matheuristics.lagrangian_relaxation_heuristic.params import LRHParams
    >>> params = LRHParams(max_iter=50, step_size=2.0)
    >>> d = params.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class LRHParams:
    """
    Configuration parameters for Lagrangian Relaxation Heuristic.

    Attributes:
        max_iter: Maximum number of subgradient iterations.
        step_size: Initial step size for subgradient method.
        halving_freq: Frequency (iterations) at which to halve the step size.
        seed: Random seed.
    """

    max_iter: int = 50
    step_size: float = 2.0
    halving_freq: int = 10
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> LRHParams:
        """Create LRHParams from a configuration object or dictionary.

        Args:
            config: Configuration object or dict with LRH parameter attributes.

        Returns:
            LRHParams: Populated parameter dataclass.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            max_iter=getattr(config, "max_iter", 50),
            step_size=getattr(config, "step_size", 2.0),
            halving_freq=getattr(config, "halving_freq", 10),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert LRHParams to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Mapping of parameter names to their values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
