"""
RL-ALNS (Reinforcement Learning-augmented Adaptive Large Neighborhood Search) configuration.

Attributes:
    RLALNSConfig: Configuration for the RL-ALNS policy.

Example:
    >>> from configs.policies.rl_alns import RLALNSConfig
    >>> config = RLALNSConfig()
    >>> config.max_iterations
    1000
    >>> config.profit_aware_operators
    False
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from logic.src.configs.policies.helpers.learning.reinforcement_learning import RLConfig
from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig


@dataclass
class RLALNSConfig(ALNSConfig):
    """Configuration for Reinforcement Learning-augmented ALNS (RL-ALNS) policy.

    Attributes:
        rl_config: Centralized Reinforcement Learning configuration.
    """

    rl_config: RLConfig = field(default_factory=RLConfig)
    profit_aware_operators: bool = False
    time_limit: float = 60.0
    max_iterations: int = 5000
    start_temp: float = 100.0
    cooling_rate: float = 0.995
    min_removal: int = 1
    max_removal_pct: float = 0.3
    seed: Optional[int] = None


    @property
    def rl_algorithm(self) -> str:
        """Name of the RL algorithm being used.

        Returns:
            str: Name of the RL algorithm.
        """
        # For ALNS, we might use both bandits or TD agents
        if self.rl_config.agent_type == "td_learning":
            return self.rl_config.td_learning.algorithm
        return self.rl_config.bandit.algorithm

    @property
    def alpha(self) -> float:
        """Learning rate for TD agents.

        Returns:
            float: Learning rate for TD agents.
        """
        return self.rl_config.td_learning.alpha

    @property
    def gamma(self) -> float:
        """Discount factor for TD agents.

        Returns:
            float: Discount factor for TD agents.
        """
        return self.rl_config.td_learning.gamma

    @property
    def epsilon(self) -> float:
        """Exploration probability for TD agents.

        Returns:
            float: Exploration probability for TD agents.
        """
        return self.rl_config.td_learning.epsilon

    @property
    def ucb_c(self) -> float:
        """UCB exploration parameter.

        Returns:
            float: UCB exploration parameter.
        """
        return self.rl_config.bandit.c

    @property
    def reward_new_global_best(self) -> float:
        """Reward for finding a new global best solution.

        Returns:
            float: Reward for finding a new global best solution.
        """
        return self.rl_config.reward.best_reward

    @property
    def reward_improved_current(self) -> float:
        """Reward for improving the current solution.

        Returns:
            float: Reward for improving the current solution.
        """
        return self.rl_config.reward.local_reward

    @property
    def reward_accepted_worse(self) -> float:
        """Reward for accepting a worsening solution (SA).

        Returns:
            float: Reward for accepting a worsening solution (SA).
        """
        return self.rl_config.reward.accepted_reward

    @property
    def reward_rejected(self) -> float:
        """Penalty/reward for a rejected solution.

        Returns:
            float: Penalty/reward for a rejected solution.
        """
        return self.rl_config.reward.rejected_reward

    @property
    def adaptive_rewards(self) -> bool:
        """Whether rewards are scaled dynamically.

        Returns:
            bool: Whether rewards are scaled dynamically.
        """
        return self.rl_config.reward.adaptive_rewards

    @property
    def normalize_rewards(self) -> bool:
        """Whether to normalize rewards across operators.

        Returns:
            bool: Whether to normalize rewards across operators.
        """
        return self.rl_config.reward.normalize_rewards

    @property
    def epsilon_decay(self) -> float:
        """Decay factor for epsilon exploration.

        Returns:
            float: Decay factor for epsilon exploration.
        """
        if self.rl_config.agent_type == "td_learning":
            return self.rl_config.td_learning.epsilon_decay
        return self.rl_config.bandit.epsilon_decay

    @property
    def epsilon_min(self) -> float:
        """Minimum value for epsilon exploration.

        Returns:
            float: Minimum value for epsilon exploration.
        """
        if self.rl_config.agent_type == "td_learning":
            return self.rl_config.td_learning.epsilon_min
        return self.rl_config.bandit.epsilon_min

    @property
    def ucb_gamma(self) -> float:
        """Discount factor for Discounted UCB.

        Returns:
            float: Discount factor for Discounted UCB.
        """
        return self.rl_config.bandit.gamma

    @property
    def ucb_window_size(self) -> int:
        """Sliding window size for SW-UCB.

        Returns:
            int: Sliding window size for SW-UCB.
        """
        return self.rl_config.bandit.window_size

    @property
    def ts_alpha_prior(self) -> float:
        """Alpha prior for Thompson Sampling.

        Returns:
            float: Alpha prior for Thompson Sampling.
        """
        return self.rl_config.bandit.alpha_prior

    @property
    def ts_beta_prior(self) -> float:
        """Beta prior for Thompson Sampling.

        Returns:
            float: Beta prior for Thompson Sampling.
        """
        return self.rl_config.bandit.beta_prior

    @property
    def exp3_gamma(self) -> float:
        """Learning rate for EXP3 agents.

        Returns:
            float: Learning rate for EXP3 agents.
        """
        return self.rl_config.bandit.gamma

    @property
    def progress_thresholds(self) -> list[float]:
        """Thresholds for classifying search progress.

        Returns:
            list[float]: Thresholds for classifying search progress.
        """
        return self.rl_config.features.progress_thresholds

    @property
    def stagnation_thresholds(self) -> list[int]:
        """Thresholds for classifying search stagnation.

        Returns:
            list[int]: Thresholds for classifying search stagnation.
        """
        return self.rl_config.features.stagnation_thresholds

    @property
    def diversity_thresholds(self) -> list[float]:
        """Thresholds for classifying population diversity.

        Returns:
            list[float]: Thresholds for classifying population diversity.
        """
        return self.rl_config.features.diversity_thresholds

    @classmethod
    def from_config(cls, config: RLALNSConfig) -> "RLALNSConfig":
        """
        Create RLALNSConfig from a RLALNSConfig dataclass.
        Note: This remains for backward compatibility with Hydra loaders.

        Args:
            config (RLALNSConfig): The configuration dataclass.

        Returns:
            "RLALNSConfig": The parameters object.
        """
        return cls(
            time_limit=config.time_limit,
            max_iterations=config.max_iterations,
            start_temp=config.start_temp,
            cooling_rate=config.cooling_rate,
            min_removal=config.min_removal,
            max_removal_pct=config.max_removal_pct,
            profit_aware_operators=getattr(config, "profit_aware_operators", False),
            rl_config=config.rl_config,
        )
