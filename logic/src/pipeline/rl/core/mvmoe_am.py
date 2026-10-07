"""
MVMoE (Multi-View Mixture of Experts) models.

Thin wrappers around POMO and REINFORCE that inject MoE-specific kwargs
into the AttentionModelPolicy, enabling expert-specialized encoding and
decoding for improved routing performance.

Reference:
    Zhou, J., Wu, Y., Song, Q., Ma, S., & Cao, J. (2024).
    MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts.
    arXiv:2405.01029.

Attributes:
    MVMoE_AM: MVMoE with Attention Model backbone.

Example:
    >>> from logic.src.pipeline.rl.core import MVMoE_AM
    >>> from logic.src.envs import COEnv
    >>> from logic.src.models import COPolicy
    >>> env = COEnv()
    >>> agent = COPolicy(env)
    >>> mvmoe_am = MVMoE_AM(env, agent)
    >>> mvmoe_am
    MVMoE_AM(env=<COEnv>, policy=<COPolicy>, baseline='rollout', actor_optimizer='adam', actor_lr=0.0001, critic_optimizer='adam', critic_lr=0.001, entropy_coef=0.01, value_loss_coef=0.5, normalize_advantage=True, enable_checkpointing=True)
"""

from __future__ import annotations

from typing import Optional

from torch import nn

from logic.src.constants.models import DEFAULT_MOE_KWARGS
from logic.src.models.core.attention_model import AttentionModelPolicy
from logic.src.pipeline.rl.core.reinforce import REINFORCE


class MVMoE_AM(REINFORCE):
    """
    MVMoE with Attention Model backbone.

    Combines MoE-enhanced AttentionModelPolicy with standard REINFORCE
    and a rollout baseline.

    Reference:
        Zhou et al. (2024). MVMoE: Multi-Task Vehicle Routing Solver
        with Mixture-of-Experts. arXiv:2405.01029.

    Attributes:
        policy: Policy network.
        moe_kwargs: Keyword arguments for MoE.
        baseline: Baseline method.
    """

    def __init__(
        self,
        policy: Optional[nn.Module] = None,
        moe_kwargs: Optional[dict] = None,
        baseline: str = "rollout",
        **kwargs,
    ):
        """
        Initialize MVMoE_AM.

        Args:
            policy: Description of policy.
            moe_kwargs: Description of moe_kwargs.
            baseline: Description of baseline.
            kwargs: Description of kwargs.
        """
        if moe_kwargs is None:
            moe_kwargs = DEFAULT_MOE_KWARGS

        if policy is None:
            env_name = kwargs.get("env_name", "ptp")
            policy = AttentionModelPolicy(
                env_name=env_name,
                moe_kwargs=moe_kwargs,
                **{k: v for k, v in kwargs.pop("policy_kwargs", {}).items()},
            )

        super().__init__(
            policy=policy,
            baseline=baseline,
            **kwargs,
        )
