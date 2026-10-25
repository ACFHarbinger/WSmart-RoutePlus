"""DACT Model for iterative routing improvement.

This module implements the `DACT` wrapper (Ma et al. 2021), a Dual Aspect
Collaborative Transformer designed for iterative improvement of VRP solutions.
It leverages collaborative encoders to process both node and position features.

Attributes:
    DACT: Primary training wrapper for the DACT policy.

Example:
    >>> from logic.src.models.core.dact.model import DACT
    >>> model = DACT(env=my_env)
    >>> out = model(td, my_env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import DACTPolicy


class DACT(nn.Module):
    """DACT Improvement Model.

    Wraps a `DACTPolicy` to provide high-level training and inference interfaces.
    Specializes in learning local search operators that improve existing solutions
    through sequential modifications.

    Attributes:
        env (RL4COEnvBase): Environment for dynamics and rewards.
        baseline (str): baseline type for REINFORCE.
        policy (DACTPolicy): The neural improvement operator.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[DACTPolicy] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        baseline: str = "rollout",
        **kwargs: Any,
    ) -> None:
        """Initializes the DACT model.

        Args:
            env: Environment managing problem physics and rewards.
            policy: Optional pre-instantiated DACT policy.
            policy_kwargs: Hyper-parameters for policy setup.
            baseline: RL baseline type (default: "rollout").
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.env = env
        self.baseline = baseline

        policy_kwargs = policy_kwargs or {}
        if policy is None:
            self.policy = DACTPolicy(env_name=env.name, **policy_kwargs)
        else:
            self.policy = policy

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "test",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Performs iterative improvement steps.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            phase: Current execution phase ("train", "val", "test").
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: map containing 'reward', 'log_likelihood', etc.
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
        """Computes REINFORCE loss for the improvement trajectory.

        Args:
            td: Initial problem state.
            batch: Data batch.
            policy_out: Raw policy outputs.
            reward: Optional reward override.
            log_likelihood: Optional log-prob override.

        Returns:
            Dict[str, Any]: Updated policy output containing the 'loss'.
        """
        reward_val = reward if reward is not None else policy_out["reward"]
        log_p = log_likelihood if log_likelihood is not None else policy_out["log_likelihood"]

        # Baseline subtraction to reduce variance
        advantage = reward_val - reward_val.mean()
        loss = -(advantage.detach() * log_p).mean()

        policy_out["loss"] = loss
        return policy_out

    def shared_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        phase: str,
    ) -> Dict[str, Any]:
        """Universal execution step for Lightning-style pipelines.

        Args:
            batch: Input data batch.
            batch_idx: Current batch index.
            phase: Current mode ('train', 'val').

        Returns:
            Dict[str, Any]: Fully processed results including loss if training.
        """
        td = self.env.reset(batch)
        out = self.policy(td, self.env, phase=phase)

        if phase == "train":
            out = self.calculate_loss(td, batch, out)

        return out
