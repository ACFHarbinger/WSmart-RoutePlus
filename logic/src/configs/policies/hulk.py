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
from typing import Any, List, Optional


@dataclass
class HULKConfig:
    """Hydra config for HULK hyper-heuristic.

    Attributes:
        seed (Optional[int]): Seed for the random number generator.
        max_iterations (int): Maximum number of iterations.
        time_limit (float): Time limit in seconds.
        restarts (int): Number of restarts.
        restart_threshold (int): Threshold for restarting.
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

    @classmethod
    def from_config(cls, config: Any) -> "HULKConfig":
        """
        Create parameters from a configuration object.

        Args:
            config: Configuration object.

        Returns:
            HULKConfig: Instance of HULKConfig.
        """
        return cls(
            max_iterations=getattr(config, "max_iterations", 1000),
            time_limit=getattr(config, "time_limit", 300.0),
            restarts=getattr(config, "restarts", 3),
            restart_threshold=getattr(config, "restart_threshold", 100),
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            epsilon=getattr(config, "epsilon", 0.3),
            epsilon_decay=getattr(config, "epsilon_decay", 0.995),
            min_epsilon=getattr(config, "min_epsilon", 0.05),
            memory_size=getattr(config, "memory_size", 50),
            accept_worse_prob=getattr(config, "accept_worse_prob", 0.1),
            acceptance_decay=getattr(config, "acceptance_decay", 0.99),
            start_temp=getattr(config, "start_temp", 100.0),
            cooling_rate=getattr(config, "cooling_rate", 0.99),
            min_temp=getattr(config, "min_temp", 0.01),
            min_destroy_size=getattr(config, "min_destroy_size", 2),
            max_destroy_pct=getattr(config, "max_destroy_pct", 0.3),
            local_search_iterations=getattr(config, "local_search_iterations", 10),
            local_search_operators=getattr(config, "local_search_operators", ["2-opt", "3-opt", "swap", "relocate"]),
            unstring_operators=getattr(config, "unstring_operators", ["type_i", "type_ii", "type_iii", "type_iv"]),
            string_operators=getattr(config, "string_operators", ["type_i", "type_ii", "type_iii", "type_iv"]),
            score_alpha=getattr(config, "score_alpha", 10.0),
            score_beta=getattr(config, "score_beta", 5.0),
            score_gamma=getattr(config, "score_gamma", -1.0),
            score_delta=getattr(config, "score_delta", 20.0),
            weight_learning_rate=getattr(config, "weight_learning_rate", 0.1),
            weight_decay=getattr(config, "weight_decay", 0.95),
        )

    def __post_init__(self):
        """
        Initialize operator weights if not provided.

        This method is called after __init__ and initializes the operator_weights
        dictionary if it is empty. The weights are initialized to 1.0 for all
        operators defined in unstring_operators, string_operators, and local_search_operators.
        """
        if not self.operator_weights:
            all_ops = self.unstring_operators + self.string_operators + self.local_search_operators
            self.operator_weights = {op: 1.0 for op in all_ops}
