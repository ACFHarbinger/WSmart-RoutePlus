"""
RL-HVPL (Reinforcement Learning Hybrid Volleyball Premier League) configuration for Hydra.

Attributes:
    RLHVPLConfig: Configuration for the RL-HVPL policy.

Example:
    >>> from configs.policies.rl_hvpl import RLHVPLConfig
    >>> config = RLHVPLConfig()
    >>> config.n_teams
    10
    >>> config.time_limit
    60.0
    >>> config.max_iterations
    100
    >>> config.profit_aware_operators
    False
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from logic.src.configs.policies.acceptance_criteria import AcceptanceConfig, BoltzmannAcceptanceConfig
from logic.src.configs.policies.helpers.learning.reinforcement_learning import RLConfig
from logic.src.configs.policies.routing.meta_heuristics.aco_ks import KSparseACOConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig


@dataclass
class RLHVPLConfig:
    """
    Configuration for the Reinforcement Learning Hybrid Volleyball Premier League policy.

    Combines VPL population dynamics, ACO initialization with Q-Learning,
    and ALNS local search with SARSA for adaptive operator selection.

    Attributes:
        n_teams: Number of teams in the VPL population.
        time_limit: Time limit for the algorithm in seconds.
        max_iterations: Maximum number of iterations.
        elite_coaching_iterations: Maximum number of iterations for elite coaching.
        regular_coaching_iterations: Maximum number of iterations for regular coaching.
        elite_size: Size of the elite set.
        sub_rate: Rate of subproblem solving.
        pheromone_update_strategy: Strategy for updating pheromones.
        profit_weight: Weight for profit in pheromone updates.
        aco: Configuration for ACO initialization.
        alns: Configuration for ALNS local search.
        rl_config: Configuration for the RL policy.
        seed: Random seed.
        profit_aware_operators: Whether to use profit-aware operators.
        packing: List of mandatory nodes.
        route_improvement: List of route improvements.
    """

    # General parameters
    n_teams: int = 10
    time_limit: float = 60.0
    max_iterations: int = 100
    elite_coaching_iterations: int = 300
    regular_coaching_iterations: int = 100
    elite_size: int = 3
    sub_rate: float = 0.2
    pheromone_update_strategy: str = "profit"
    profit_weight: float = 1.0

    # Nested component configs
    aco: KSparseACOConfig = field(default_factory=KSparseACOConfig)
    alns: ALNSConfig = field(default_factory=ALNSConfig)

    # RL Configuration (Centralized)
    rl_config: RLConfig = field(default_factory=RLConfig)

    # Common policy fields
    seed: Optional[int] = None
    profit_aware_operators: bool = False
    packing: List[str] = field(default_factory=list)
    route_improvement: List[Any] = field(default_factory=list)
    aco_params: KSparseACOConfig = field(
        default_factory=lambda: KSparseACOConfig(
            n_ants=10,
            k_sparse=10,
            alpha=1.0,
            beta=2.0,
            rho=0.1,
            scale=5.0,
            tau_0=None,
            tau_min=0.001,
            tau_max=10.0,
            max_iterations=1,  # Single construction per call
            time_limit=10.0,
            local_search=True,
            local_search_iterations=50,
            elitist_weight=1.0,
        )
    )
    alns_params: ALNSConfig = field(
        default_factory=lambda: ALNSConfig(
            max_iterations=200,
            acceptance_criterion=AcceptanceConfig(
                method="bmc", params=BoltzmannAcceptanceConfig(initial_temp=100.0, alpha=0.97, start_temp_control=0.05)
            ),
            reaction_factor=0.5,
            xi=0.3,
            time_limit=20.0,
        )
    )

    def __post_init__(self) -> None:
        """Sync flags across sub-parameters.

        Args:
            None.

        Returns:
            None
        """
        if self.aco_params:
            self.aco_params.profit_aware_operators = self.profit_aware_operators
        if self.alns_params:
            self.alns_params.profit_aware_operators = self.profit_aware_operators

    def _get_val(self, category: str, key: str, default: Any = None) -> Any:
        """Get value from RL configuration.

        Args:
            category (str): Category of the parameter.
            key (str): Key of the parameter.
            default (Any): Default value.

        Returns:
            Any: Value of the parameter.
        """
        # Resolve the category (e.g. td_learning, sarsa)
        if isinstance(self.rl_config, dict):
            cat_obj = self.rl_config.get(category, {})
        else:
            cat_obj = getattr(self.rl_config, category, None)

        # If no category found but we asked for sarsa, fallback to td_learning
        if not cat_obj and category == "sarsa":
            return self._get_val("td_learning", key, default)

        if isinstance(cat_obj, dict):
            return cat_obj.get(key, default)
        return getattr(cat_obj, key, default)

    def _get_param(self, key: str, default: Any = None) -> Any:
        """Get parameter from RL configuration.

        Args:
            key (str): Key of the parameter.
            default (Any): Default value.

        Returns:
            Any: Value of the parameter.
        """
        if isinstance(self.rl_config, dict):
            return self.rl_config.get("params", {}).get(key, default)
        return self.rl_config.params.get(key, default)

    @property
    def qlearning_alpha(self) -> float:
        """Learning rate for Q-Learning.

        Returns:
            float: Learning rate for Q-Learning.
        """
        return float(self._get_val("td_learning", "alpha", 0.1))

    @property
    def qlearning_gamma(self) -> float:
        """Discount factor for Q-Learning.

        Returns:
            float: Discount factor for Q-Learning.
        """
        return float(self._get_val("td_learning", "gamma", 0.95))

    @property
    def qlearning_epsilon(self) -> float:
        """Exploration rate for Q-Learning.

        Returns:
            float: Exploration rate for Q-Learning.
        """
        return float(self._get_val("td_learning", "epsilon", 0.1))

    @property
    def qlearning_epsilon_decay(self) -> float:
        """Decay factor for Q-Learning exploration rate.

        Returns:
            float: Decay factor for Q-Learning exploration rate.
        """
        return float(self._get_val("td_learning", "epsilon_decay", 0.995))

    @property
    def qlearning_epsilon_decay_step(self) -> int:
        """Frequency of Q-Learning exploration rate decay (in iterations).

        Returns:
            int: Frequency of Q-Learning exploration rate decay (in iterations).
        """
        return self._get_val("td_learning", "epsilon_decay_step", 20)

    @property
    def qlearning_epsilon_min(self) -> float:
        """Minimum exploration rate for Q-Learning.

        Returns:
            float: Minimum exploration rate for Q-Learning.
        """
        return float(self._get_val("td_learning", "epsilon_min", 0.05))

    @property
    def qlearning_improvement_thresholds(self) -> Tuple[float, float]:
        """Thresholds for classifying solution quality improvements in Q-Learning.

        Returns:
            Tuple[float, float]: Thresholds for classifying solution quality improvements in Q-Learning.
        """
        val = self._get_param("qlearning_improvement_thresholds", (1e-4, 1e-2))
        return float(val[0]), float(val[1])

    @property
    def qlearning_history_size(self) -> int:
        """Size of the rolling history window for Q-Learning state estimation.

        Returns:
            int: Size of the rolling history window for Q-Learning state estimation.
        """
        return self._get_param("qlearning_history_size", 50)

    @property
    def sarsa_alpha(self) -> float:
        """Learning rate for SARSA (used in ALNS).

        Returns:
            float: Learning rate for SARSA (used in ALNS).
        """
        return float(self._get_val("sarsa", "alpha", 0.1))

    @property
    def sarsa_gamma(self) -> float:
        """Discount factor for SARSA.

        Returns:
            float: Discount factor for SARSA.
        """
        return float(self._get_val("sarsa", "gamma", 0.95))

    @property
    def sarsa_epsilon(self) -> float:
        """Exploration rate for SARSA.

        Returns:
            float: Exploration rate for SARSA.
        """
        return float(self._get_val("sarsa", "epsilon", 0.1))

    @property
    def sarsa_epsilon_decay(self) -> float:
        """Decay factor for SARSA exploration rate.

        Returns:
            float: Decay factor for SARSA exploration rate.
        """
        return float(self._get_val("sarsa", "epsilon_decay", 0.995))

    @property
    def sarsa_epsilon_decay_step(self) -> int:
        """Frequency of SARSA exploration rate decay (in iterations).

        Returns:
            int: Frequency of SARSA exploration rate decay (in iterations).
        """
        return self._get_val("sarsa", "epsilon_decay_step", 20)

    @property
    def sarsa_epsilon_min(self) -> float:
        """Minimum exploration rate for SARSA.

        Returns:
            float: Minimum exploration rate for SARSA.
        """
        return float(self._get_val("sarsa", "epsilon_min", 0.05))

    @property
    def sarsa_diversity_size(self) -> int:
        """Maximum number of diverse solutions to track for SARSA reward calculation.

        Returns:
            int: Maximum number of diverse solutions to track for SARSA reward calculation.
        """
        return self._get_param("sarsa_diversity_size", 50)

    @property
    def sarsa_improvement_thresholds(self) -> Tuple[float, float]:
        """Thresholds for solution quality improvement rewards in SARSA.

        Returns:
            Tuple[float, float]: Thresholds for solution quality improvement rewards in SARSA.
        """
        val = self._get_param("sarsa_improvement_thresholds", (1e-4, 1e-2))
        return float(val[0]), float(val[1])

    @property
    def sarsa_operator_progress_thresholds(self) -> Tuple[float, float]:
        """Progress-based thresholds for adjusting SARSA rewards.

        Returns:
            Tuple[float, float]: Progress-based thresholds for adjusting SARSA rewards.
        """
        val = self._get_param("sarsa_operator_progress_thresholds", (0.3, 0.7))
        return float(val[0]), float(val[1])

    @property
    def sarsa_operator_stagnation_thresholds(self) -> Tuple[int, int]:
        """Stagnation count thresholds for triggering SARSA penalty/reset.

        Returns:
            Tuple[int, int]: Stagnation count thresholds for triggering SARSA penalty/reset.
        """
        val = self._get_param("sarsa_operator_stagnation_thresholds", (10, 50))
        return int(val[0]), int(val[1])

    @property
    def sarsa_operator_diversity_thresholds(self) -> Tuple[float, float]:
        """Diversity-based thresholds for adjusting SARSA rewards.

        Returns:
            Tuple[float, float]: Diversity-based thresholds for adjusting SARSA rewards.
        """
        val = self._get_param("sarsa_operator_diversity_thresholds", (0.2, 0.5))
        return float(val[0]), float(val[1])
