"""
DR-GRPO (GRPO done Right) algorithm.

Reference:
    Zheng, G., Chen, W., Chadha, A., & Ren, X. (2025). Understanding R1-Zero-Like
    Training: A Critical Perspective. arXiv preprint arXiv:2503.20783.
    https://arxiv.org/abs/2503.20783

Attributes:
    DRGRPO: GRPO done Right algorithm.

Example:
    >>> from logic.src.pipeline.learning.core import DRGRPO
    >>> from logic.src.pipeline.learning.envs import DRGRPOEnv
    >>> from logic.src.models import DRGRPOPPOAgent
    >>> env = DRGRPOEnv()
    >>> agent = DRGRPOPPOAgent(env)
    >>> dr_grpo = DRGRPO(env, agent)
    >>> dr_grpo
    DRGRPO(env=<DRGRPOEnv>, policy=<DRGRPOPPOAgent>, baseline='rollout', baseline_kwargs={'val_dataset': 'val', 'update_every': 10, 'use_rollouts': True}, actor_optimizer='adam', actor_lr=0.0001, critic_optimizer='adam', critic_lr=0.001, entropy_coef=0.01, value_loss_coef=0.5, normalize_advantage=True, enable_checkpointing=True)
"""

from __future__ import annotations

import torch

from logic.src.pipeline.learning.core.ppo import PPO


class DRGRPO(PPO):
    """
    DR-GRPO (GRPO done Right).

    Features:
    - Zero-mean centered advantages without std normalization: A = R - Mean(R_group).
    - No sequence length normalization in the objective function.

    Reference:
        Zheng, G., Chen, W., Chadha, A., & Ren, X. (2025). Understanding R1-Zero-Like
        Training: A Critical Perspective. arXiv:2503.20783.
        https://arxiv.org/abs/2503.20783

    Attributes:
        env: The environment instance used for training.
        agent: The PPO agent for DR-GRPO.
        instance_generator: The generator used to create new problem instances.
        lr: Learning rate for the optimizer.
        gamma: Discount factor for future rewards.
        gae_lambda: Factor for generalized advantage estimation.
        clip_epsilon: Clipping parameter for PPO.
        value_loss_coef: Coefficient for the value function loss.
        entropy_coef: Coefficient for the entropy bonus.
        max_grad_norm: Maximum gradient norm for clipping.
        inner_epochs: Number of epochs to train on collected data.
        n_steps_per_epoch: Number of steps to collect per epoch.
        batch_size: Batch size for training updates.
    """

    def calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        DR-GRPO centered advantages.

        Args:
            rewards: Tensor of rewards.
            values: Tensor of state values.

        Returns:
            Tensor of advantages.
        """
        # GRPO uses group mean as baseline
        # In this implementation, we treat the batch as the group
        advantage = rewards - rewards.mean()
        return advantage

    def calculate_ratio(self, new_log_p: torch.Tensor, old_log_p: torch.Tensor) -> torch.Tensor:
        """
        Importance ratio (no sequence length normalization).

        Args:
            new_log_p: Tensor of new log probabilities.
            old_log_p: Tensor of old log probabilities.

        Returns:
            Tensor of importance ratios.
        """
        return torch.exp(new_log_p - old_log_p.detach())
