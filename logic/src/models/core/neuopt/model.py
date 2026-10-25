"""Neural Optimizer (NeuOpt) Model.

This module provides the `NeuOpt` wrapper (Ma et al. 2019), which leverages
neural networks to optimize combinatorial solutions by identifying and
applying high-quality local search moves.

Attributes:
    NeuOpt: Primary training and inference wrapper for the Neural Optimizer.

Example:
    >>> from logic.src.models.core.neuopt.model import NeuOpt
    >>> model = NeuOpt(embed_dim=128, num_layers=3)
    >>> out = model(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import NeuOptPolicy


class NeuOpt(nn.Module):
    """NeuOpt Model wrapper for neural iterative optimization.

    Coordinates the `NeuOptPolicy` to perform refinement passes on existing
    trajectories. It encodes the current solution topology and decodes a
    sequence of improvements.

    Attributes:
        policy (NeuOptPolicy): The underlying neural optimization policy.
        env (Optional[RL4COEnvBase]): Environment managing problem context.
    """

    def __init__(
        self,
        env: Optional[RL4COEnvBase] = None,
        policy: Optional[NeuOptPolicy] = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        **policy_kwargs: Any,
    ) -> None:
        """Initializes the NeuOpt model.

        Args:
            env: Environment managing problem logic and step transitions.
            policy: Optional pre-initialized NeuOptPolicy instance.
            embed_dim: Feature width for embeddings and internal layers.
            num_heads: Number of attention heads in the transformer layers.
            num_layers: Count of encoder-decoder blocks.
            policy_kwargs: Additional arguments for NeuOptPolicy.
        """
        super().__init__()
        if policy is None:
            policy = NeuOptPolicy(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                **policy_kwargs,
            )
        self.policy = policy
        self.env = env

    def forward(
        self,
        td: Any,
        env: Optional[RL4COEnvBase] = None,
        strategy: str = "greedy",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes the neural optimization pass.

        Args:
            td: TensorDict or dict containing problem state and initial solution.
            env: Optional environment override.
            strategy: Decoding strategy ("greedy" or "sampling").
            kwargs: Additional keyword arguments for the policy.

        Returns:
            Dict[str, Any]: Optimization results including refined actions and log-likelihoods.
        """
        env = env or self.env
        return self.policy(td, env, strategy=strategy, **kwargs)
