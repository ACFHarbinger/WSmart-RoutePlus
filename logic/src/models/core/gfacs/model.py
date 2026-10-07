"""GFACS Model for probabilistic routing.

This module implements `GFACS` (GFlowNet Ant Colony System), which treats tour
construction as a flow network (Ye et al. 2024). It optimizes using Trajectory
Balance (TB) loss by matching forward flows with backward flows under uniform
action priors.

Attributes:
    GFACS: Primary training wrapper for GFlowNet-based Ant Colony System.

Example:
    >>> from logic.src.models.core.gfacs.model import GFACS
    >>> model = GFACS(env=my_env)
    >>> out = model(td, phase="train")
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, cast

import numpy as np
import scipy.special
import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import GFACSPolicy


class GFACS(nn.Module):
    """GFlowNet Ant Colony System (GFACS).

    Learns a generative policy for routing problems using Trajectory Balance.
    The model integrates ant-based exploration with GFlowNet's ability
    to sample solutions proportional to their rewards.

    Attributes:
        env (RL4COEnvBase): Targeted optimization environment.
        policy (GFACSPolicy): Neural flow model.
        train_with_local_search (bool): If True, incorporates LS-refined rewards.
        alpha_min (float): Initial weight for local search in loss.
        alpha_max (float): maximum weight for local search in loss.
        alpha_flat_epochs (int): Epochs before alpha scheduling begins.
        beta_min (float): Initial reward-to-likelihood scaling.
        beta_max (float): Maximum reward-to-likelihood scaling.
        beta_flat_epochs (int): Epochs before beta scheduling begins.
        current_epoch (int): Counter for hyperparameter scheduling.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[GFACSPolicy] = None,
        baseline: str = "no",
        train_with_local_search: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        baseline_kwargs: Optional[Dict[str, Any]] = None,
        alpha_min: float = 0.5,
        alpha_max: float = 1.0,
        alpha_flat_epochs: int = 5,
        beta_min: float = 1.0,
        beta_max: float = 1.0,
        beta_flat_epochs: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initializes the GFACS model.

        Args:
            env: Environment managing states and rewards.
            policy: Optional custom GFACS policy.
            baseline: Baseline type identifier (e.g., "no", "rollout").
            train_with_local_search: Whether to incorporate local search in training.
            policy_kwargs: Hyper-parameters for the underlying policy.
            baseline_kwargs: Configuration for the baseline model.
            alpha_min: Starting weight for local search reward advantage.
            alpha_max: Final weight for local search reward advantage.
            alpha_flat_epochs: Epochs to hold alpha at alpha_min.
            beta_min: Starting reward-to-likelihood scaling exponent.
            beta_max: Final reward-to-likelihood scaling exponent.
            beta_flat_epochs: Epochs to hold beta at beta_min.
            kwargs: Additional keyword arguments.
        """
        super().__init__()

        policy_kwargs = policy_kwargs or {}
        if policy is None:
            policy = GFACSPolicy(
                env_name=env.name,
                train_with_local_search=train_with_local_search,
                **policy_kwargs,
            )

        self.env = env
        self.policy = policy
        self.train_with_local_search = train_with_local_search

        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_flat_epochs = alpha_flat_epochs
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_flat_epochs = beta_flat_epochs

        self.current_epoch = 0

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        phase: str = "train",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Samples trajectories using the GFlowNet policy.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            phase: Current execution phase ("train", "val", "test").
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Policy results including actions and rewards.
        """
        env = env or self.env
        return self.policy(td, env, phase=phase, **kwargs)

    @property
    def alpha(self) -> float:
        """Scheduled weight for local-search reward inclusion.

        Returns:
            float: Interpolated alpha value.
        """
        if not hasattr(self, "trainer") or self.trainer is None or not hasattr(self.trainer, "max_epochs"):
            return self.alpha_min
        return self.alpha_min + (self.alpha_max - self.alpha_min) * min(
            self.current_epoch / (cast(Any, self.trainer).max_epochs - self.alpha_flat_epochs),
            1.0,
        )

    @property
    def beta(self) -> float:
        """Scheduled exponent for reward-to-flow mapping.

        Returns:
            float: Interpolated beta value.
        """
        if not hasattr(self, "trainer") or self.trainer is None or not hasattr(self.trainer, "max_epochs"):
            return self.beta_min
        return self.beta_min + (self.beta_max - self.beta_min) * min(
            math.log(self.current_epoch + 1) / math.log(cast(Any, self.trainer).max_epochs - self.beta_flat_epochs),
            1.0,
        )

    def calculate_loss(
        self,
        td: TensorDict,
        batch: TensorDict,
        policy_out: Dict[str, Any],
        reward: Optional[torch.Tensor] = None,
        log_likelihood: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes Trajectory Balance loss (TB).

        Args:
            td: Problem state.
            batch: Data batch.
            policy_out: Results from dynamic rollout.
            reward: Optional reward override.
            log_likelihood: Optional likelihood override.

        Returns:
            torch.Tensor: mean squared TB error.
        """
        reward_val = policy_out["reward"]
        n_ants = reward_val.size(1)
        advantage = reward_val - reward_val.mean(dim=1, keepdim=True)

        if self.train_with_local_search:
            ls_reward = policy_out["ls_reward"]
            ls_advantage = ls_reward - ls_reward.mean(dim=1, keepdim=True)
            weighted_advantage = advantage * (1 - self.alpha) + ls_advantage * self.alpha
        else:
            weighted_advantage = advantage
            ls_advantage = torch.zeros_like(advantage)

        # TB condition: Log(Z) + Forward_Likelihood == Reward + Backward_Prior
        forward_flow = policy_out["log_likelihood"] + policy_out["logZ"].repeat(1, n_ants)
        backward_flow = (
            self.calculate_log_pb_uniform(policy_out["actions"], n_ants) + weighted_advantage.detach() * self.beta
        )
        tb_loss = torch.pow(forward_flow - backward_flow, 2).mean()

        if self.train_with_local_search:
            ls_forward_flow = policy_out["ls_log_likelihood"] + policy_out["ls_logZ"].repeat(1, n_ants)
            ls_backward_flow = (
                self.calculate_log_pb_uniform(policy_out["ls_actions"], n_ants) + ls_advantage.detach() * self.beta
            )
            ls_tb_loss = torch.pow(ls_forward_flow - ls_backward_flow, 2).mean()
            tb_loss = tb_loss + ls_tb_loss

        return tb_loss

    def calculate_log_pb_uniform(self, actions: torch.Tensor, n_ants: int) -> torch.Tensor:
        """Calculates uniform backward probability for the trajectory.

        Args:
            actions: sampled node indices [B*Ants, N].
            n_ants: population size.

        Returns:
            torch.Tensor: Log prior of generating the path in reverse.
        """
        if self.env.name == "tsp":
            return torch.tensor(math.log(1 / (2 * actions.size(1))))
        elif self.env.name == "cvrp":
            _a1 = actions.detach().cpu().numpy()
            n_nodes = np.count_nonzero(_a1, axis=1)
            _a2 = _a1[:, 1:] - _a1[:, :-1]
            n_routes = np.count_nonzero(_a2, axis=1) - n_nodes
            _a3 = _a1[:, 2:] - _a1[:, :-2]
            n_multinode_routes = np.count_nonzero(_a3, axis=1) - n_nodes
            log_b_p = -scipy.special.gammaln(n_routes + 1) - n_multinode_routes * math.log(2)
            from logic.src.utils.decoding import unbatchify

            return unbatchify(torch.from_numpy(log_b_p).to(actions.device), n_ants)
        elif self.env.name in ("op", "pctsp", "ptp"):
            return torch.tensor(math.log(1 / 2))
        else:
            raise ValueError(f"Unknown environment: {self.env.name}")
