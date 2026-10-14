"""
Pure Island Model GA configuration.

Replaces SLC with rigorous GA terminology.

Attributes:
    MemeticAlgorithmIslandModelConfig: Configuration for the Memetic Algorithm Island Model policy.

Example:
    >>> from configs.policies.ma_im import MemeticAlgorithmIslandModelConfig
    >>> config = MemeticAlgorithmIslandModelConfig()
    >>> config.max_generations
    50
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MemeticAlgorithmIslandModelConfig:
    """Configuration for Memetic Algorithm Island Model (MA-IM) policy.

    Functionally equivalent to SLC.

    Attributes:
        n_islands (int): Number of islands in the GA.
        island_size (int): Number of individuals per island.
        max_generations (int): Maximum number of generations to run.
        stagnation_limit (int): Number of generations without improvement before triggering stagnation handling.
        n_removal (int): Number of individuals to remove during replacement.
        local_search_iterations (int): Number of local search iterations per individual.
        time_limit (float): Time limit in seconds.
        seed (Optional[int]): Random seed.
        vrpp (bool): Whether the problem is a VRRP.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        packing (Optional[List[PackingConfig]]): Mandatory customers/requests selection strategies.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement strategies.
    """

    n_islands: int = 5
    island_size: int = 4
    max_generations: int = 50
    stagnation_limit: int = 5
    n_removal: int = 1
    local_search_iterations: int = 500
    time_limit: float = 60.0
    seed: Optional[int] = None
    vrpp: bool = True
    profit_aware_operators: bool = False
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
