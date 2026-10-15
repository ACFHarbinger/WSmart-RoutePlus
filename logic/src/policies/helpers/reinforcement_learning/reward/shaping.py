"""Reward shaping strategies for reinforcement learning in optimization search.

This module provides mechanisms to transform objective value improvements into
reinforcement signals, with support for adaptive reward scaling.

Attributes:
    RewardShaper: Base class for converting search outcomes into reward signals.
    AdaptiveRewardShaper: Reward shaper that adapts reward values based on search progress.

Example:
    >>> shaper = RewardShaper(best_improvement_reward=10.0)
    >>> reward = shaper.compute_reward(new_profit=100.0, prev_profit=90.0, ...)
"""

from typing import Dict


class RewardShaper:
    """
    Handles reward shaping for reinforcement learning-based optimization policies.

    Converts changes in solution quality (profit or cost) into standardized
    reward signals for RL agents. This abstraction allows agents to learn from
    different optimization outcomes such as finding a new best solution,
    local improvements, or simple acceptance of a non-improving step.

    Attributes:
        best_reward (float): Reward for finding a new globally best solution.
        local_reward (float): Reward for improving over the current local solution.
        accepted_reward (float): Base reward for an accepted step (even if non-improving).
        rejected_reward (float): Penalty for a rejected candidate solution.
        stagnation_penalty (float): Scalable penalty applied when search fails to improve.
    """

    def __init__(
        self,
        best_improvement_reward: float = 10.0,
        local_improvement_reward: float = 5.0,
        accepted_reward: float = 1.0,
        rejected_reward: float = -1.0,
        stagnation_penalty: float = -0.1,
    ):
        """Initialize the reward shaper.

        Args:
            best_improvement_reward: Reward for new best solution.
            local_improvement_reward: Reward for local improves.
            accepted_reward: Reward for accepted operators.
            rejected_reward: Penalty for rejected moves.
            stagnation_penalty: Step-wise penalty for stagnation.
        """

        self.best_reward = best_improvement_reward
        self.local_reward = local_improvement_reward
        self.accepted_reward = accepted_reward
        self.rejected_reward = rejected_reward
        self.stagnation_penalty = stagnation_penalty

    def compute_reward(
        self,
        new_profit: float,
        prev_profit: float,
        best_profit: float,
        accepted: bool,
        stagnation_count: int = 0,
        improvement_threshold: float = 1e-6,
    ) -> float:
        """Compute shaped reward based on the outcome of a search step.

        Logic hierarchy:
        1. New Global Best -> best_reward
        2. Local Improvement -> local_reward
        3. Acceptance (no improve) -> accepted_reward + (stagnation * penalty)
        4. Rejection -> rejected_reward

        Args:
            new_profit: Total profit of the newly generated solution.
            prev_profit: Total profit of the solution before the current step.
            best_profit: Maximum profit found so far in the search.
            accepted: Boolean indicating if the move was accepted by the policy.
            stagnation_count: Number of iterations since the last global best improvement.
            improvement_threshold: Minimum epsilon for considering a change an 'improvement'.

        Returns:
            float: The calculated float reward for the RL agent.
        """
        # Step 1: Check for global improvement
        # If the new profit is significantly better than the best profit found so far,
        # it's a new global best.
        if new_profit > best_profit + improvement_threshold:
            return self.best_reward

        # Step 2: Check for local improvement
        # If the new profit is significantly better than the previous profit,
        # it's a local improvement.
        if new_profit > prev_profit + improvement_threshold:
            return self.local_reward

        # Step 3: Handle accepted but non-improving moves
        # If the move was accepted but didn't lead to a global or local improvement,
        # apply the base accepted reward and a penalty for stagnation.
        if accepted:
            # Apply a penalty that grows with the duration of stagnation
            penalty = stagnation_count * self.stagnation_penalty
            return self.accepted_reward + penalty

        # Step 4: Move was rejected entirely
        # If none of the above conditions met, the move was rejected.
        return self.rejected_reward

    def calculate_reward(
        self, new_cost: float, current_cost: float, best_cost: float, accepted: bool, stagnation_count: int = 0
    ) -> float:
        """Legacy/Backward compatibility method for cost-minimization problems.

        Inverts the logic of compute_reward since lower cost is better.

        Args:
            new_cost: Cost of the new solution.
            current_cost: Cost of the current solution.
            best_cost: Minimum cost found so far.
            accepted: Acceptance flag.
            stagnation_count: Stagnation duration.

        Returns:
            float: Shaped reward value.
        """
        # Step 1: Global best check (minimization)
        # If the new cost is significantly lower than the best cost found so far,
        # it's a new global best.
        if new_cost < best_cost - 1e-6:
            return self.best_reward
        # Step 2: Local improvement check (minimization)
        # If the new cost is significantly lower than the current cost,
        # it's a local improvement.
        if new_cost < current_cost - 1e-6:
            return self.local_reward
        # Step 3: Acceptance check
        # If the move was accepted but didn't lead to a global or local improvement,
        # apply the base accepted reward and a penalty for stagnation.
        if accepted:
            return self.accepted_reward + stagnation_count * self.stagnation_penalty
        # Step 4: Move was rejected entirely
        # If none of the above conditions met, the move was rejected.
        return self.rejected_reward

    def get_reward_config(self) -> Dict[str, float]:
        """Returns the current reward configuration.

        Returns:
            Dict[str, float]: A dictionary containing the current reward values.
        """
        return {
            "best": self.best_reward,
            "local": self.local_reward,
            "accepted": self.accepted_reward,
            "rejected": self.rejected_reward,
        }


class AdaptiveRewardShaper(RewardShaper):
    """Adapts rewards based on search progress.

    Attributes:
        adaptation_rate: Rate at which rewards adapt to search progress.
        base_rewards: Dictionary of initial reward values.
    """

    def __init__(self, *args, adaptation_rate: float = 0.5, **kwargs):
        """Initialize the adaptive reward shaper.

        Args:
            args: Positional arguments passed to RewardShaper.
            adaptation_rate: Rate at which rewards adapt to search progress.
            kwargs: Keyword arguments passed to RewardShaper.
        """

        super().__init__(*args, **kwargs)
        self.adaptation_rate = adaptation_rate
        self.base_rewards = {"best": self.best_reward, "local": self.local_reward, "accepted": self.accepted_reward}

    def adapt(self, progress: float):
        """Adapt reward values based on current search progress.

        Boosts acceptance rewards early in the search to encourage exploration,
        and boosts improvement rewards late in the search for exploitation.

        Args:
            progress: Search completion progress in [0.0, 1.0].

        """

        # Higher exploration early, higher exploitation late
        exploration_boost = (1.0 - progress) * self.adaptation_rate
        self.accepted_reward = self.base_rewards["accepted"] * (1.0 + exploration_boost)

        exploitation_boost = progress * self.adaptation_rate
        self.best_reward = self.base_rewards["best"] * (1.0 + exploitation_boost)
        self.local_reward = self.base_rewards["local"] * (1.0 + exploitation_boost)

    def compute_reward(self, *args, progress: float = 0.5, **kwargs) -> float:
        """Adapts rewards and computes the shaped value.

        Args:
            args: Positional arguments for RewardShaper.compute_reward.
            progress: Search completion progress in [0.0, 1.0].
            kwargs: Keyword arguments for RewardShaper.compute_reward.


        Returns:
            float: The adapted and shaped reward value.
        """

        self.adapt(progress)
        return super().compute_reward(*args, **kwargs)
