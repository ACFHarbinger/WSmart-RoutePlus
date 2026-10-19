"""
Soft Adaptive Policy Optimization (SAPO) algorithm.

Reference:
    Gao, C., Zheng, C., et al. (2025). Soft Adaptive Policy Optimization.
    arXiv preprint arXiv:2511.20347.
    https://arxiv.org/abs/2511.20347

Attributes:
    SAPO: Soft Adaptive Policy Optimization.

Example:
    >>> from logic.src.pipeline.rl.core import SAPO
    >>> from logic.src.pipeline.rl.envs import COEnv
    >>> from logic.src.models import COPolicy
    >>> env = COEnv()
    >>> agent = COPolicy(env)
    >>> sapo = SAPO(env, agent)
    >>> sapo
    SAPO(env=<COEnv>, policy=<COPolicy>, baseline='rollout', actor_optimizer='adam', actor_lr=0.0001, critic_optimizer='adam', critic_lr=0.001, entropy_coef=0.01, value_loss_coef=0.5, normalize_advantage=True, enable_checkpointing=True, tau_pos=0.1, tau_neg=1.0)
"""

from __future__ import annotations

import torch

from logic.src.pipeline.rl.core.ppo import PPO


class SAPO(PPO):
    """
    Soft Adaptive Policy Optimization (SAPO).

    SAPO adaptively adjusts clipping based on advantage sign:
    - Uses soft gates instead of hard clipping: f(r) = (4/tau) * sigmoid(tau * (r-1))
    - tau_pos for positive advantages (more aggressive updates)
    - tau_neg for negative advantages (more conservative updates)

    Reference:
        Gao, C., Zheng, C., et al. (2025). Soft Adaptive Policy Optimization.
        arXiv:2511.20347. https://arxiv.org/abs/2511.20347

    Attributes:
        tau_pos: Positive advantage clipping coefficient.
        tau_neg: Negative advantage clipping coefficient.
    """

    def __init__(
        self,
        tau_pos: float = 0.1,
        tau_neg: float = 1.0,
        **kwargs,
    ):
        """
        Initialize SAPO module.

        Args:
            tau_pos: Positive advantage clipping coefficient (default: 0.1).
            tau_neg: Negative advantage clipping coefficient (default: 1.0).
            kwargs: Additional arguments to pass to the parent class (PPO).
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["critic", "env", "policy", "kwargs", "generator"])
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def calculate_actor_loss(self, ratio: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
        """
        SAPO soft-gated objective.

        Args:
            ratio: Importance ratios.
            advantage: Estimated advantages.

        Returns:
            The SAPO loss.
        """
        # Adaptive tau selection based on advantage sign
        tau = torch.where(advantage > 0, self.tau_pos, self.tau_neg)

        # Soft gate function replaces hard clipping in PPO
        # f(r) = (4/tau) * sigmoid(tau * (r-1))
        f_ratio = (4.0 / tau) * torch.sigmoid(tau * (ratio - 1.0))

        # SAPO objective: E[f(r) * A]
        return -(f_ratio * advantage).mean()
