"""
Configuration parameters for the Kernel Search (KS) matheuristic.

Attributes:
    KSParams: Dataclass holding Kernel Search solver hyperparameters.

Example:
    >>> from logic.src.policies.routing.matheuristics.kernel_search.params import KSParams
    >>> params = KSParams(initial_kernel_size=50, time_limit=300.0)
    >>> d = params.to_dict()
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class KSParams:
    """
    Configuration parameters for Kernel Search.

    Attributes:
        initial_kernel_size: Size of the starting variable pool.
        bucket_size: Size for search buckets.
        max_buckets: Limit on improvement attempts.
        time_limit: Total time budget for optimization.
        mip_limit_nodes: Node limit for internal Gurobi solves.
        mip_gap: Acceptable relative optimality gap.
        seed: Random seed.
    """

    initial_kernel_size: int = 50
    bucket_size: int = 20
    max_buckets: int = 15
    time_limit: float = 300.0
    mip_limit_nodes: int = 10000
    mip_gap: float = 0.01
    seed: int = 42
    engine: str = "gurobi"
    framework: str = "ortools"

    @classmethod
    def from_config(cls, config: Any) -> KSParams:
        """Create KSParams from a configuration object or dictionary.

        Args:
            config: Configuration object or dict with KS parameter attributes.

        Returns:
            KSParams: Populated parameter dataclass.
        """
        if isinstance(config, dict):
            return cls(**{k: v for k, v in config.items() if k in {f.name for f in fields(cls)}})

        return cls(
            initial_kernel_size=getattr(config, "initial_kernel_size", 50),
            bucket_size=getattr(config, "bucket_size", 20),
            max_buckets=getattr(config, "max_buckets", 15),
            time_limit=getattr(config, "time_limit", 300.0),
            mip_limit_nodes=getattr(config, "mip_limit_nodes", 10000),
            mip_gap=getattr(config, "mip_gap", 0.01),
            seed=getattr(config, "seed", 42),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert KSParams to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Mapping of parameter names to their values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
