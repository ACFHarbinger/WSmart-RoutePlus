"""NARGNN Model: Non-Autoregressive GNN for routing.

This module provides the `NARGNN` wrapper, which employs a edge-based Graph
Neural Network to predict a heatmap of edge inclusion probabilities. This
heatmap is then used to construct solutions in a single non-autoregressive
pass (or via sampling), with the model refined using REINFORCE.

Attributes:
    NARGNN: Training wrapper for edge-based heatmap models.

Example:
    >>> model = NARGNN(env_name="tsp", embed_dim=64)
    >>> out = model(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import NARGNNPolicy


class NARGNN(nn.Module):
    """NARGNN training wrapper for REINFORCE.

    Assembles the `NARGNNPolicy` and computes standard reinforcement learning
    losses against various baseline strategies to handle the discrete
    construction rewards.

    Attributes:
        policy (NARGNNPolicy): The underlying heatmap-prediction GNN policy.
        baseline_type (Optional[str]): Method for advantage calculation.
        _baseline_val (Optional[torch.Tensor]): State for 'exponential' baseline.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        env_name: str = "tsp",
        num_layers_heatmap_generator: int = 5,
        num_layers_graph_encoder: int = 15,
        baseline: Optional[str] = "rollout",
        **kwargs: Any,
    ) -> None:
        """Initializes the NARGNN wrapper.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            env_name: Name of the environment identifier.
            num_layers_heatmap_generator: Number of layers in the MLP heatmap generator.
            num_layers_graph_encoder: Number of edge-based GNN encoder layers.
            baseline: RL baseline type ("rollout", "exponential", "no").
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.policy = NARGNNPolicy(
            embed_dim=embed_dim,
            env_name=env_name,
            num_layers_heatmap_generator=num_layers_heatmap_generator,
            num_layers_graph_encoder=num_layers_graph_encoder,
            **kwargs,
        )
        self.baseline_type = baseline
        self._baseline_val: Optional[torch.Tensor] = None

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes heatmap generation and computes REINFORCE loss.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Result map including reward, likelihood, and 'loss'.
        """
        out = self.policy(td, env, **kwargs)

        reward = out["reward"]
        log_likelihood = out["log_likelihood"]

        # Compute baseline statefully or per-batch
        if self.baseline_type == "exponential":
            if self._baseline_val is None:
                self._baseline_val = reward.mean().detach()
            else:
                self._baseline_val = 0.8 * self._baseline_val + 0.2 * reward.mean().detach()
            baseline = self._baseline_val
        elif self.baseline_type == "rollout":
            baseline = reward.mean()
        else:
            baseline = 0.0

        # Loss calculation: REINFORCE advantage
        advantage = reward - baseline
        loss = -(advantage.detach() * log_likelihood).mean()
        out["loss"] = loss
        out["baseline"] = baseline

        return out

    def set_strategy(self, strategy: str, **kwargs: Any) -> None:
        """Configures the action selection tactic (e.g., 'greedy', 'sampling').

        Args:
            strategy: Decoding strategy identifier (e.g., "sampling").
            kwargs: Additional keyword arguments.
        """
        self.policy.set_strategy(strategy, **kwargs)
