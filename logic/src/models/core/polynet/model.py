"""PolyNet Model: Learning Diverse Solution Strategies.

This module provides the `PolyNet` wrapper, which learns a set of $K$ diverse
routing strategies using binary vector conditioning. It implements the "Poppy"
loss, which uses the instance-wide mean reward as a competitive baseline to
encourage strategy specialization.

Attributes:
    PolyNet: Diversity-focused training wrapper with Poppy loss.

Example:
    >>> model = PolyNet(env, k=128)
    >>> out = model(td, env, phase="train")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, cast

import torch
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader

from logic.src.envs.base.base import RL4COEnvBase

from .policy import PolyNetPolicy


class PolyNet(nn.Module):
    """PolyNet Model for population-based RL.

    Encourages a single model to exhibit multiple distinct behaviors by
    conditioning on a strategy index. Training utilizes the Poppy loss to
    maximize the collective performance of the population.

    Attributes:
        env (RL4COEnvBase): Environment for states and rewards.
        k (int): Number of distinct strategies to learn.
        val_num_solutions (int): Candidate count for strategy selection during val.
        num_augment (int): Data augmentation factor.
        policy (PolyNetPolicy): The underlying strategy-conditioned policy.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[PolyNetPolicy] = None,
        k: int = 128,
        val_num_solutions: int = 800,
        encoder_type: str = "AM",
        policy_kwargs: Optional[Dict[str, Any]] = None,
        num_augment: int = 8,
        **kwargs: Any,
    ) -> None:
        """Initializes the PolyNet model.

        Args:
            env: Environment managing states and rewards.
            policy: Optional custom PolyNet policy.
            k: Number of strategies in the population.
            val_num_solutions: Solution candidates for validation.
            encoder_type: Base model architecture (e.g., "AM").
            policy_kwargs: Hyper-parameters for the underlying policy.
            num_augment: Data augmentation factor for training.
            kwargs: Additional keyword arguments.
        """
        super().__init__()

        policy_kwargs = policy_kwargs or {}
        self.env = env
        self.k = k
        self.val_num_solutions = val_num_solutions
        self.num_augment = num_augment

        if policy is None:
            self.policy = PolyNetPolicy(
                k=k,
                encoder_type=encoder_type,
                env_name=env.name,
                **policy_kwargs,
            )
        else:
            self.policy = policy

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "train",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Routes execution to the strategy-conditioned policy.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            phase: Current execution phase ("train", "val", "test").
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Results containing rewards and log probabilities.
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
        """Performs a unified execution step for various pipeline phases.

        Args:
            batch: Batch of problem instances.
            batch_idx: Global index of the batch.
            phase: pipeline state ('train', 'val', 'test').

        Returns:
            Dict[str, Any]: Metrics and outputs from the step.
        """
        td = self.env.reset(batch)
        num_solutions = self.k if phase == "train" else self.val_num_solutions

        # Augmentation logic: expand batch via symmetric transformations
        if phase == "train" and self.num_augment > 1:
            td = cast(TensorDict, self.env.reset(batch)).repeat_interleave(self.num_augment, dim=0)

        out = self.policy(
            td=td,
            env=self.env,
            phase=phase,
            return_actions=True,
            num_starts=num_solutions,
        )

        if phase == "train":
            out = self.calculate_loss(td, batch, out)

        return out

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Computes the PolyNet Poppy loss.

        Poppy loss defines the advantage relative to the mean performance of the
        entire strategy population for a given instance, forcing each strategy
        to find better local optima than the global average.

        Args:
            td: Problem instance container.
            batch: Data batch (meta-information).
            policy_out: Result map from the policy forward pass.
            reward: Optional override for the multi-path reward tensor.
            log_likelihood: Optional override for the path log-probabilities.

        Returns:
            Dict[str, Any]: result updated with 'loss' and 'max_reward'.
        """
        reward = reward if reward is not None else policy_out["reward"]
        log_likelihood = log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]

        # Ensure multi-path shape: [Batch, K]
        if reward.dim() == 1:
            from logic.src.utils.decoding import unbatchify

            reward = unbatchify(reward, self.k)
            log_likelihood = unbatchify(log_likelihood, self.k)

        # Shared instance-wise baseline (mean of population)
        baseline = reward.mean(dim=-1, keepdim=True)
        advantage = reward - baseline

        # Scalar loss for the population
        loss = -(advantage * log_likelihood).mean()
        max_reward = reward.max(dim=-1).values.mean()

        policy_out.update(
            {
                "loss": loss,
                "max_reward": max_reward,
                "baseline": baseline.mean(),
            }
        )
        return policy_out

    def rollout(
        self,
        dataset: Any,
        batch_size: int = 64,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Evaluates the strategy population across a full dataset.

        Args:
            dataset: input instances.
            batch_size: inference width.
            device: hardware target.

        Returns:
            torch.Tensor: Best reward per instance [DatasetLength].
        """
        self.eval()
        self.to(device)

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=getattr(dataset, "collate_fn", None),
        )

        rewards = []
        for batch in dl:
            with torch.inference_mode():
                batch = self.env.reset(batch.to(device))
                result = self.policy(batch, self.env, phase="val")
                rewards.append(result["reward"].max(dim=-1).values)

        return torch.cat(rewards, dim=0)
