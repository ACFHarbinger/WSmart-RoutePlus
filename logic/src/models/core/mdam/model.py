"""Multi-Decoder Attention Model (MDAM) for diverse routing.

This module provides the `MDAM` wrapper, which trains multiple independent
decoders sharing a common encoder. It employs a specialized REINFORCE scheme
where the baseline is the maximum reward across all decoders, encouraging
diversity through KL-divergence regularization.

Attributes:
    MDAM: Diverse multi-policy training wrapper.

Example:
    >>> model = MDAM(env=vrpp_env, kl_weight=0.01)
    >>> out = model(td)
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Optional, Union

import torch
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader

from logic.src.envs.base.base import RL4COEnvBase

from .policy import MDAMPolicy


def mdam_rollout(
    baseline_self: Any,
    model: MDAM,
    env: RL4COEnvBase,
    batch_size: int = 64,
    device: Union[str, torch.device] = "cpu",
    dataset: Any = None,
) -> torch.Tensor:
    """Specialized rollout for MDAM that selects the best path per instance.

    Computes the maximum reward across all decoder paths to provide a tight
    baseline for the REINFORCE advantage.

    Args:
        baseline_self: Instance of the baseline object being patched.
        model: MDAM architecture instance.
        env: Target problem environment.
        batch_size: Batch size for inference passes.
        device: Hardware device to execute on.
        dataset: Input data for rollout evaluation.

    Returns:
        torch.Tensor: Best-path rewards per instance [Batch].
    """
    dataset = baseline_self.dataset if dataset is None else dataset

    model.eval()
    model = model.to(device)

    def eval_model(batch: TensorDict) -> torch.Tensor:
        """Evaluate the model on a batch of data.

        Args:
            batch: Data container.

        Returns:
            Best rewards found across decoders.
        """
        with torch.inference_mode():
            batch = env.reset(batch.to(device))
            result = model(batch, env, strategy="greedy")
            # Baseline is the best realization among decoders
            return result["reward"].max(dim=1).values

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=getattr(dataset, "collate_fn", None),
    )

    rewards = torch.cat([eval_model(batch) for batch in dl], dim=0)
    return rewards


class MDAM(nn.Module):
    """MDAM training wrapper with Multi-Path Baseline.

    Implements the diverse training strategy from Xin et al. (AAAI 2021).
    Encourages population-style coverage of the solution space.

    Attributes:
        env (RL4COEnvBase): Environment for states and rewards.
        kl_weight (float): Scalar weighting for the diversity entropy term.
        policy (MDAMPolicy): The underlying multi-decoder neural policy.
        baseline_type (str): Meta-identifier for RL baseline logic.
        baseline (Optional[Any]): Concrete baseline object (set during training).
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[MDAMPolicy] = None,
        baseline: str = "rollout",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        baseline_kwargs: Optional[Dict[str, Any]] = None,
        kl_weight: float = 0.01,
    ) -> None:
        """Initializes the MDAM model.

        Args:
            env: Targeted problem environment.
            policy: Pre-instantiated MDAM policy.
            baseline: Identifier for the RL baseline method.
            policy_kwargs: Dictionary for default policy setup.
            baseline_kwargs: Dictionary for baseline parameters.
            kl_weight: Diversity regularization strength.
        """
        super().__init__()

        policy_kwargs = policy_kwargs or {}
        self.env = env
        self.kl_weight = kl_weight

        if policy is None:
            self.policy = MDAMPolicy(env_name=env.name, **policy_kwargs)
        else:
            self.policy = policy

        self.baseline_type = baseline
        self.baseline_kwargs = baseline_kwargs or {}
        self.baseline: Optional[Any] = None

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "train",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Routes execution to the underlying MDAM policy.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            phase: Current execution phase ("train", "val", "test").
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Multi-path construction results.
        """
        if env is None:
            env = self.env
        return self.policy(td, env, phase=phase, **kwargs)

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Computes REINFORCE loss with competitive multi-path baseline.

        Incorporates KL-divergence to penalize policy collapse toward a single
        mode, ensuring decoder diversity.

        Args:
            td: current problem state.
            batch: Data batch (may contain pre-computed baseline).
            policy_out: Map containing 'reward', 'log_likelihood', and 'entropy'.
            reward: Optional override for the reward tensor.
            log_likelihood: Optional override for the log_likelihood tensor.

        Returns:
            Dict[str, Any]: Updated result map including the scalar 'loss'.
        """
        extra = batch.get("extra", None)
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]
        kl_divergence = policy_out.get("entropy", torch.tensor(0.0, device=td.device))

        # Evaluate baseline
        if self.baseline is not None and extra is None:
            bl_val, bl_loss = self.baseline.eval(td, reward, self.env)
        else:
            bl_val = extra if extra is not None else torch.tensor(0.0, device=reward.device)
            bl_loss = torch.tensor(0.0, device=reward.device)

        # Broadcast baseline [B] to multi-path reward [B, NumDecoders]
        if isinstance(bl_val, torch.Tensor) and len(bl_val.shape) == 1:
            bl_val = bl_val.unsqueeze(1)

        # Advantage based on the best path found so far
        advantage = reward - bl_val
        reinforce_loss = -(advantage * log_likelihood).mean()

        # Final loss: REINFORCE + Baseline + Entropy Regularization
        # Subtracting KL weight encourages increased entropy (diversity)
        loss = reinforce_loss + bl_loss - self.kl_weight * kl_divergence

        policy_out.update(
            {
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
                "kl_loss": kl_divergence,
            }
        )
        return policy_out

    @staticmethod
    def patch_baseline_rollout(baseline: Any) -> None:
        """Injects MDAM-aware multi-path logic into a standard RL baseline.

        Overwrites the evaluation rollout of the baseline object to use the
        competitive max-reward strategy.

        Args:
            baseline: RL baseline instance to modify.
        """
        if hasattr(baseline, "baseline"):
            inner = baseline.baseline
            if hasattr(inner, "rollout"):
                inner.rollout = partial(mdam_rollout, inner)
        elif hasattr(baseline, "rollout"):
            baseline.rollout = partial(mdam_rollout, baseline)
