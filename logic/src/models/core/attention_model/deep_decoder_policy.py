"""Deep Decoder Policy for constructive routing.

This module provides a policy that utilizes a multi-layer (deep) attention
decoder for constructing routing solutions. This architecture allows for
more complex decision-making during decoding compared to single-layer variants.

Attributes:
    DeepDecoderPolicy: Constructive policy with a multi-layer graph decoder.

Example:
    >>> from logic.src.models.core.attention_model.deep_decoder_policy import DeepDecoderPolicy
    >>> policy = DeepDecoderPolicy(env_name="ptp", n_decode_layers=3)
    >>> out = policy(td, env)
"""

from __future__ import annotations

from typing import Any

from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.models.subnets.decoders.gat import DeepGATDecoder


class DeepDecoderPolicy(AttentionModelPolicy):
    """Routing policy with multi-layer attention construction.

    Inherits the autoregressive decoding pipeline from AttentionModelPolicy
    while employing a DeepGATDecoder, which applies multiple layers of
    cross-attention between the current state and node embeddings at each
    decoding step.

    Attributes:
        encoder (GraphAttentionEncoder): Graph feature extractor.
        decoder (DeepGATDecoder): Multi-layer sequential constructor.
        init_embedding (nn.Module): Problem-specific latent projection.
    """

    decoder: DeepGATDecoder

    def __init__(
        self,
        env_name: str,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        n_encode_layers: int = 3,
        n_decode_layers: int = 3,
        n_heads: int = 8,
        normalization: str = "batch",
        dropout_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initializes the DeepDecoderPolicy.

        Args:
            env_name: Name of the environment identifier.
            embed_dim: Dimensionality of node features.
            hidden_dim: Dimensionality of hidden layers.
            n_encode_layers: Number of Transformer encoder layers.
            n_decode_layers: Number of Transformer decoder layers.
            n_heads: Number of attention heads.
            normalization: Type of normalization ("batch", "layer", "instance").
            dropout_rate: Dropout probability.
            kwargs: Additional keyword arguments.
        """
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_encode_layers=n_encode_layers,
            n_heads=n_heads,
            normalization=normalization,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        self.decoder = DeepGATDecoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_decode_layers,
            normalization=normalization,
            dropout_rate=dropout_rate,
            **kwargs,
        )
