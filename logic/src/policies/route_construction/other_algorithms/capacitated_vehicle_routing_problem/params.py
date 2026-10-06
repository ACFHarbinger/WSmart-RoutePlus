"""
Configuration parameters for the Capacitated Vehicle Routing Problem (CVRP) policy.

Attributes:
    CVRPParams: Configuration parameters for the CVRP solver.

Example:
    >>> from logic.src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem import CVRPParams
    >>> params = CVRPParams()
    >>> params
    CVRPParams(engine='ortools', time_limit=2.0, seed=42)
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class CVRPParams:
    """
    Configuration parameters for the CVRP solver.

    Attributes:
        engine: Optimization engine to use ('ortools', 'pyvrp' or 'clarke_wright').
        time_limit: Time limit for the solver in seconds.
        seed: Random seed for reproducibility.
    """

    engine: str = "ortools"
    time_limit: float = 2.0
    seed: int = 42

    @classmethod
    def from_config(cls, config: Any) -> CVRPParams:
        """Create CVRPParams from a configuration object or dictionary.

        Args:
            config: Configuration object or dictionary.

        Returns:
            CVRPParams: Configured CVRP parameters.
        """
        if isinstance(config, dict):
            values = {k: v for k, v in config.items() if k in {f.name for f in fields(cls)} and v is not None}
            return cls(**values)

        seed = getattr(config, "seed", None)
        return cls(
            engine=getattr(config, "engine", "ortools"),
            time_limit=getattr(config, "time_limit", 2.0),
            seed=42 if seed is None else seed,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert CVRPParams to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of CVRPParams.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
