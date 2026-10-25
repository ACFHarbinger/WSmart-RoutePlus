"""Neural Neighborhood Search (N2S) Model.

This module provides the `N2S` wrapper (Li et al. 2021), which uses neural
networks to guide neighborhood search for combinatorial optimization. It
specializes in learning to select promising local moves to improve solutions.

Attributes:
    N2S: Primary Training and inference wrapper for Neural Neighborhood Search.

Example:
    >>> from logic.src.models.core.n2s.model import N2S
    >>> model = N2S(embed_dim=128, k_neighbors=20)
    >>> out = model(td, env)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch import nn

from logic.src.envs.base.base import RL4COEnvBase

from .policy import N2SPolicy


class N2S(nn.Module):
    """N2S Model wrapper for neural local search.

    Orchestrates the `N2SPolicy` to iteratively improve solutions. The model
    identifies candidate neighbors and selects the most promising one to
    apply as a local move.

    Attributes:
        policy (N2SPolicy): The underlying neural neighborhood search policy.
        env (Optional[RL4COEnvBase]): Environment managing problem dynamics.
    """

    def __init__(
        self,
        env: Optional[RL4COEnvBase] = None,
        policy: Optional[N2SPolicy] = None,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **policy_kwargs: Any,
    ) -> None:
        """Initializes the N2S model.

        Args:
            env: Optional environment for state transitions.
            policy: Optional pre-instantiated N2SPolicy.
            embed_dim: Dimensionality of node features.
            num_heads: Number of attention heads.
            k_neighbors: Number of candidate neighbors to consider per node.
            policy_kwargs: Additional arguments for N2SPolicy.
        """
        super().__init__()
        if policy is None:
            policy = N2SPolicy(
                embed_dim=embed_dim,
                num_heads=num_heads,
                k_neighbors=k_neighbors,
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
        """Executes the improvement pass.

        Args:
            td: TensorDict containing current solution state.
            env: Optional environment override.
            strategy: Decoding strategy (e.g., "greedy", "sampling").
            kwargs: Additional keyword arguments for the policy.

        Returns:
            Dict[str, Any]: Improvement results containing actions and weights.
        """
        env = env or self.env
        return self.policy(td, env, strategy=strategy, **kwargs)
