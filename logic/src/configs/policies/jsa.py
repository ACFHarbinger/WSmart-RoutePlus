"""
Joint Simulated Annealing Configuration for Hydra.

Attributes:
    JointSAConfig: Configuration for the Joint Simulated Annealing policy.

Example:
    >>> from configs.policies.jsa import JointSAConfig
    >>> config = JointSAConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class JointSAConfig:
    """
    Hydra configuration for the Joint Simulated Annealing policy.

    Attributes:
        start_temp (float): Initial temperature.
        cooling_rate (float): Rate of temperature cooling.
        max_steps (int): Maximum number of iterations.
        restart_limit (int): Maximum number of restarts.
        prob_bit_flip (float): Probability of bit flip operation.
        prob_route_swap (float): Probability of route swap operation.
        overflow_penalty (float): Penalty for overflow.
        seed (Optional[int]): Random seed for reproducibility.
        time_limit (float): Time limit in seconds.
        packing (Optional[List[Any]]): Mandatory customers/requests selection.
        route_improvement (Optional[List[Any]]): Route improvement strategies.
    """

    start_temp: float = 1000.0
    cooling_rate: float = 0.995
    max_steps: int = 2000
    restart_limit: int = 5
    prob_bit_flip: float = 0.7
    prob_route_swap: float = 0.3
    overflow_penalty: float = 1000.0
    seed: Optional[int] = 42
    time_limit: float = 60.0

    # Hydra compatibility placeholders
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
