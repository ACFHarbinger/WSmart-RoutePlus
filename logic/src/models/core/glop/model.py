"""GLOP Model for large-scale routing.

This module implements `GLOP` (Global-Local Optimization), a hierarchical
approach that learns to partition a large problem into smaller sub-problems
(Meta-partitioning) which are then solved by local constructive solvers.

Attributes:
    GLOP: Training orchestrator for the hierarchical partitioning model.

Example:
    >>> from logic.src.models.core.glop.model import GLOP
    >>> model = GLOP(env=large_scale_env, policy_kwargs={'n_samples': 4})
    >>> out = model(td)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import GLOPPolicy


class GLOP(nn.Module):
    """Global-Local Optimization Policy (GLOP).

    Trains a neural partitioner using REINFORCE. The model learns to cluster
    nodes into manageable segments such that the total cost after local
    construction and merge is minimized.

    Attributes:
        env (RL4COEnvBase): Environment instance for state transitions.
        policy (GLOPPolicy): Neural partitioner and local solver wrapper.
        baseline (str): Reinforcement learning baseline (default: "mean").
        n_samples (int): count of parallel partition candidates per instance.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[GLOPPolicy] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        baseline: str = "mean",
        **kwargs: Any,
    ) -> None:
        """Initializes the GLOP model.

        Args:
            env: Environment managing problem physics.
            policy: Optional pre-initialized GLOP policy.
            policy_kwargs: Hyper-parameters for the underlying policy.
            baseline: RL baseline type (e.g., "mean").
            kwargs: Additional keyword arguments.
        """
        super().__init__()

        policy_kwargs = policy_kwargs or {}
        self.env = env
        self.baseline = baseline

        if policy is None:
            self.policy = GLOPPolicy(env_name=env.name, **policy_kwargs)
        else:
            self.policy = policy

        self.n_samples = self.policy.n_samples

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "train",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes the partitioning and local optimization pipeline.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            phase: Current execution phase ("train", "val", "test").
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: results including 'reward', 'log_likelihood', and 'actions'.
        """
        if env is None:
            env = self.env
        return self.policy(td, env, phase=phase, **kwargs)

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> Dict[str, Any]:
        """Standardized execution step for Lightning-style pipelines.

        Args:
            batch: Data container with problem instances.
            batch_idx: Sequential index of the batch.
            phase: Descriptive label for the current pass.

        Returns:
            Dict[str, Any]: execution metadata including loss and top rewards.
        """
        td = self.env.reset(batch)

        # Forward pass with action tracking
        out = self.policy(
            td=td,
            env=self.env,
            phase=phase,
            return_actions=True,
        )

        # Performance aggregation
        from logic.src.utils.decoding import unbatchify

        reward = unbatchify(out["reward"], self.n_samples)
        max_reward, _ = reward.max(dim=-1)
        out["max_reward"] = max_reward

        # Gradient computation
        if phase == "train":
            assert self.n_samples > 1, "GLOP training requires multiple samples for baseline"
            log_likelihood = unbatchify(out["log_likelihood"], self.n_samples)

            # Standard REINFORCE with leave-one-out mean baseline
            advantage = reward - reward.mean(dim=-1, keepdim=True)
            loss = -(advantage * log_likelihood).mean()
            out["loss"] = loss

        return out

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Computes REINFORCE gradients based on policy outputs.

        Args:
            td: problem state.
            batch: input data.
            policy_out: mapping of constructed rewards and probabilities.
            reward: optional override for the reward tensor.
            log_likelihood: optional override for the likelihood tensor.

        Returns:
            Dict[str, Any]: updated policy output containing 'loss'.
        """
        reward_val = reward if reward is not None else policy_out["reward"]
        log_prob = log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]

        # Group by sample dimensions
        from logic.src.utils.decoding import unbatchify

        reward_flat = unbatchify(reward_val, self.n_samples)
        log_prob_flat = unbatchify(log_prob, self.n_samples)

        # Mean-centered policy gradient
        advantage = reward_flat - reward_flat.mean(dim=-1, keepdim=True)
        loss = -(advantage * log_prob_flat).mean()

        policy_out["loss"] = loss
        policy_out["max_reward"] = reward_flat.max(dim=-1).values.mean()

        return policy_out
