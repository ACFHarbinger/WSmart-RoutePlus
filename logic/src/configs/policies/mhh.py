"""
Configuration for the Matheuristic Hyper-Heuristic (MHH).

Attributes:
    MHHConfig: Configuration for the Matheuristic Hyper-Heuristic (MHH) policy.

Example:
    >>> from configs.policies.mhh import MHHConfig
    >>> config = MHHConfig()
    >>> config.iters
    10
"""

from dataclasses import dataclass
from typing import Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MHHConfig:
    """
    Configuration for the Matheuristic Hyper-Heuristic (MHH) policy.

    MHH applies matheuristic Low-Level Heuristics (LLHs) iteratively to
    find improvements in the multi-period plan.

    Attributes:
        iters (int): Number of iterations. Defaults to 10.
        seed (int): Random seed for reproducibility. Defaults to 42.
        packing (Optional[PackingConfig]): Configuration for
            mandatory node selection policies.
        route_improvement (Optional[RouteImprovingConfig]): Optional configuration
            for local search refinement steps.
    """

    iters: int = 10
    seed: int = 42

    packing: Optional[PackingConfig] = None
    route_improvement: Optional[RouteImprovingConfig] = None
