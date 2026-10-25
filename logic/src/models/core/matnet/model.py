"""MatNet Model for matrix-aware routing.

This module provides the `MatNet` wrapper (Kwon et al. 2021), designed for
problems with matrix-form inputs (e.g., ATSP, FFSP). It uses a REINFORCE-based
training scheme with various baseline options.

Attributes:
    MatNet: Primary wrapper for matrix-based neural combinatorial optimization.

Example:
    >>> from logic.src.models.core.matnet.model import MatNet
    >>> model = MatNet(embed_dim=256, baseline="rollout")
    >>> out = model(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import MatNetPolicy


class MatNet(nn.Module):
    """MatNet training wrapper for REINFORCE.

    Assembles the `MatNetPolicy` and manages the reinforcement learning baseline
    logic (exponential or rollout) to compute stable gradient updates.

    Attributes:
        policy (MatNetPolicy): The underlying matrix encoder-decoder policy.
        baseline_type (str): Identifier for the baseline method ('rollout', 'exponential').
        _baseline_val (Optional[torch.Tensor]): Internal state for exponential baseline.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 5,
        n_heads: int = 8,
        tanh_clipping: float = 10.0,
        normalization: str = "instance",
        baseline: str = "rollout",
        env_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the MatNet wrapper.

        Args:
            embed_dim: Dimensionality of latent embeddings.
            hidden_dim: Dimensionality of hidden layers.
            num_layers: Number of transformer encoder layers.
            n_heads: Number of attention heads.
            tanh_clipping: Range for logit clipping (0 to disable).
            normalization: Type of layer normalization.
            baseline: RL baseline type ("rollout", "exponential", "no").
            env_name: Name of the environment identifier.
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.policy = MatNetPolicy(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            normalization=normalization,
            problem=None,
            **kwargs,
        )
        self.baseline_type = baseline
        self._baseline_val: Optional[Any] = None

    def forward(
        self,
        td: TensorDict,
        env: Optional[RL4COEnvBase] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Performs a forward pass and computes the REINFORCE loss.

        Args:
            td: TensorDict containing problem instance data.
            env: Environment managing problem physics.
            kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Results containing rewards, actions, and calculated loss.
        """
        out = self.policy(td, env, **kwargs)

        # Ensure reward is available for RL
        if "reward" not in out and env is not None:
            out["reward"] = env.get_reward(td, out["actions"])

        if "reward" in out and "log_p" in out:
            reward = out["reward"]
            log_likelihood = out["log_p"]

            # Compute advantage against the selected baseline
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

            # Compute standard REINFORCE loss: -E[(R - b) * log_p]
            advantage = reward - baseline
            loss = -(advantage.detach() * log_likelihood).mean()
            out["loss"] = loss
            out["baseline"] = baseline

        return out

    def set_strategy(self, strategy: str, **kwargs: Any) -> None:
        """Configures the action selection tactic in the policy.

        Args:
            strategy: Decoding strategy identifier (e.g., "greedy", "sampling").
            kwargs: Additional keyword arguments.
        """
        self.policy.set_strategy(strategy, **kwargs)
