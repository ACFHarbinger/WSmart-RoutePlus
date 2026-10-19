"""shared_critic.py module.

Attributes:
    SharedBaseline: Shared baseline using a critic that shares the encoder with the actor.

Example:
    >>> from logic.src.pipeline.learning.common.baselines import SharedBaseline
    >>> baseline = SharedBaseline()
    >>> td = TensorDict({"obs": torch.randn(2, 10, 20)}, batch_size=[2])
    >>> reward = torch.tensor([1.0, 2.0])
    >>> baseline.eval(td, reward)
    tensor([1.0, 2.0])
"""

from typing import Any, Optional

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.critic_network.policy import create_critic_from_actor
from logic.src.utils.functions.rl import ensure_tensordict

from .base import Baseline


class SharedBaseline(Baseline):
    """
    Shared baseline using a critic that shares the encoder with the actor.

    Creates a critic network from the actor's encoder via weight sharing
    (deepcopy). This enables the critic to leverage the same learned
    representations while having its own value head.

    Attributes:
        critic: The shared critic network.
    """

    def __init__(
        self,
        critic: Optional[nn.Module] = None,
        **kwargs,
    ):
        """
        Initialize SharedBaseline.



        Args:
            critic: Optional pre-built critic module. If None, will be created from policy in setup().
            kwargs: Additional keyword arguments (unused).
        """
        super().__init__()
        self.critic = critic

    def setup(self, policy: nn.Module):
        """
        Build the critic from the actor's encoder.

        Args:
            policy: Actor policy with an encoder attribute.
        """
        if self.critic is not None:
            return  # Already built

        self.critic = create_critic_from_actor(policy)

    def eval(self, td: TensorDict, reward: torch.Tensor, env: Optional[Any] = None) -> torch.Tensor:  # type: ignore[override]
        """
        Compute baseline value using shared critic.

        Args:
            td: TensorDict with environment state.
            reward: Current batch rewards (fallback shape).
            env: Environment (unused).

        Returns:
            torch.Tensor: Critic value predictions.
        """
        if self.critic is None:
            return torch.zeros_like(reward)

        td = ensure_tensordict(td, next(iter(self.critic.parameters())).device)
        return self.critic(td).squeeze(-1)

    def get_learnable_parameters(self) -> list:
        """Get learnable parameters for the shared critic.

        Returns:
            List of learnable parameters.
        """
        return list(self.critic.parameters()) if self.critic is not None else []
