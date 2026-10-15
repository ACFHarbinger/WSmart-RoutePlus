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
from typing import Optional

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
