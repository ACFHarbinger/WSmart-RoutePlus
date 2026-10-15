"""
Hydra configuration for HULK hyper-heuristic.

Attributes:
    HULKConfig: Configuration for the HULK policy.

Example:
    >>> from configs.policies.hulk import HULKConfig
    >>> config = HULKConfig()
    >>> config.max_iterations
    1000
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class HULKConfig:
    """Hydra config for HULK hyper-heuristic.

    Attributes:
        seed (Optional[int]): Seed for the random number generator.
        max_iterations (int): Maximum number of iterations.
        time_limit (float): Time limit in seconds.
        restarts (int): Number of restarts.
        restart_threshold (int): Threshold for restarting.
        vrpp (bool): Whether the problem is a VRRP.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        epsilon (float): Epsilon parameter for operator selection.
        epsilon_decay (float): Decay rate for epsilon.
        min_epsilon (float): Minimum epsilon value.
        memory_size (int): Size of the performance memory.
        accept_worse_prob (float): Probability of accepting worse solutions.
        acceptance_decay (float): Decay rate for acceptance probability.
        start_temp (float): Initial temperature for simulated annealing.
        cooling_rate (float): Cooling rate for simulated annealing.
        min_temp (float): Minimum temperature for simulated annealing.
        min_destroy_size (int): Minimum number of routes to destroy.
        max_destroy_pct (float): Maximum percentage of routes to destroy.
        local_search_iterations (int): Number of local search iterations.
        local_search_operators (List[str]): List of local search operators.
        unstring_operators (List[str]): List of unstringing operators.
        string_operators (List[str]): List of stringing operators.
        score_alpha (float): Alpha parameter for scoring.
        score_beta (float): Beta parameter for scoring.
        score_gamma (float): Gamma parameter for scoring.
        score_delta (float): Delta parameter for scoring.
        weight_learning_rate (float): Learning rate for weight updates.
        weight_decay (float): Decay rate for weights.
    """

    # Search parameters
    seed: Optional[int] = None
    max_iterations: int = 1000
    time_limit: float = 300.0
    restarts: int = 3
    restart_threshold: int = 100
    vrpp: bool = True
    profit_aware_operators: bool = False

    # Operator selection
    epsilon: float = 0.3
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05

    # Performance tracking
    memory_size: int = 50

    # Acceptance
    accept_worse_prob: float = 0.1
    acceptance_decay: float = 0.99

    # Simulated annealing
    start_temp: float = 100.0
    cooling_rate: float = 0.99
    min_temp: float = 0.01

    # Destruction/repair
    min_destroy_size: int = 2
    max_destroy_pct: float = 0.3

    # Local search
    local_search_iterations: int = 10
    local_search_operators: List[str] = field(default_factory=lambda: ["2-opt", "3-opt", "swap", "relocate"])

    # Unstringing operators
    unstring_operators: List[str] = field(default_factory=lambda: ["type_i", "type_ii", "type_iii", "type_iv"])

    # Stringing operators
    string_operators: List[str] = field(default_factory=lambda: ["type_i", "type_ii", "type_iii", "type_iv"])

    # Scoring
    score_alpha: float = 10.0  # score improvement
    score_beta: float = 5.0  # score accept
    score_gamma: float = -1.0  # score reject
    score_delta: float = 20.0  # score best

    # Weight learning
    weight_learning_rate: float = 0.1
    weight_decay: float = 0.95
    score_best: float = 20.0
    score_improvement: float = 10.0
    score_accept: float = 5.0
    score_reject: float = 0.5
    operator_weights: dict = field(default_factory=dict)


def register_configs():
    """Register HULK configs with Hydra."""
    cs = ConfigStore.instance()
    cs.store(group="policies", name="hulk", node=HULKConfig)
