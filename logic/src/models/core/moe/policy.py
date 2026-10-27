"""Mixture of Experts (MoE) Policy for routing.

This module provides `MoEPolicy`, an extension of `AttentionModelPolicy` that
replaces the standard Feed-Forward Networks in the transformer encoder with
Sparse Mixture of Experts layers to increase model capacity without proportional
computational cost.

Attributes:
    MoEPolicy: Routing model with expert-routed attention encoding.

Example:
    >>> policy = MoEPolicy(env_name="tsp", embed_dim=128, num_experts=4)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any, cast

from logic.src.models.core.attention_model import AttentionModelPolicy
from logic.src.models.subnets.encoders.moe.encoder import MoEGraphAttentionEncoder


class MoEPolicy(AttentionModelPolicy):
    """Mixture of Experts Policy.

    Utilizes `MoEGraphAttentionEncoder` to route node features through a subset
    of specialized 'expert' networks during encoding, allowing the model to
    specialize on different graph structures or tasks.

    Attributes:
        encoder (MoEGraphAttentionEncoder): Sparse-routed transformer encoder.
    """

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
        num_experts: int = 4,
        k: int = 2,
        noisy_gating: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the Mixture of Experts policy.

        Args:
            env_name: Name of the environment identifier.
            embed_dim: Dimensionality of latent embeddings.
            hidden_dim: Dimensionality of hidden layers.
            n_encode_layers: Number of transformer encoder layers.
            n_heads: Number of attention heads.
            normalization: Type of layer normalization.
            num_experts: Total number of experts in the MoE layer.
            k: Number of experts to activate per token (top-k).
            noisy_gating: Whether to use noise in the gating network.
            kwargs: Additional keyword arguments.
        """
        # We invoke super() but we will overwrite the encoder.
        # This is slightly inefficient (creates standard encoder then throws it away)
        # but prevents duplicating all the other setup logic in AttentionModelPolicy.
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_encode_layers=n_encode_layers,
            n_heads=n_heads,
            normalization=normalization,
            **kwargs,
        )

        # Overwrite encoder with MoE version
        self.encoder = cast(
            Any,  # attribute is typed as the base encoder
            MoEGraphAttentionEncoder(
                n_heads=n_heads,
                embed_dim=embed_dim,
                feed_forward_hidden=hidden_dim,
                n_layers=n_encode_layers,
                normalization=normalization,
                num_experts=num_experts,
                k=k,
                noisy_gating=noisy_gating,
                **kwargs,
            ),
        )
