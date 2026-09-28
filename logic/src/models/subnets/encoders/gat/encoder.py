"""Graph Attention Encoder.

Attributes:
    GraphAttentionEncoder: Graph Attention Encoder with stacked MultiHeadAttentionLayers.

Example:
    >>> from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder
    >>> encoder = GraphAttentionEncoder(n_heads=8, embed_dim=128, n_layers=3)
"""

from __future__ import annotations

from typing import Any, Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.common import TransformerEncoderBase

from .gat_multi_head_attention_layer import GATMultiHeadAttentionLayer


class GraphAttentionEncoder(TransformerEncoderBase):
    """Graph Attention Encoder with stacked MultiHeadAttentionLayers.

    Supports standard Transformer architecture and Hyper-Networks.
    Inherits from TransformerEncoderBase for common encoder patterns.

    Attributes:
        n_sublayers (Optional[int]): Number of sublayers (unused, for API compatibility).
        agg (Any): Aggregation method (unused, for API compatibility).
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        n_layers: int,
        n_sublayers: Optional[int] = None,
        feed_forward_hidden: int = 512,
        norm_config: Optional[NormalizationConfig] = None,
        normalization: Optional[str] = None,
        activation_config: Optional[ActivationConfig] = None,
        dropout_rate: float = 0.1,
        agg: Any = None,
        connection_type: str = "skip",
        expansion_rate: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initializes the GraphAttentionEncoder.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension.
            n_layers: Number of encoder layers.
            n_sublayers: Number of sublayers (unused).
            feed_forward_hidden: Hidden dimension for feed-forward layers.
            norm_config: Normalization configuration.
            normalization: Type of normalization ('batch', 'layer', etc.).
            activation_config: Activation function configuration.
            dropout_rate: Dropout probability.
            agg: Aggregation method (unused).
            connection_type: Connection type ("skip", "dense", or "hyper").
            expansion_rate: Expansion factor for hyper-connections.
            kwargs: Additional keyword arguments.
        """
        if norm_config is None and normalization is not None:
            norm_config = NormalizationConfig(norm_type=normalization)

        super(GraphAttentionEncoder, self).__init__(
            n_heads=n_heads,
            embed_dim=embed_dim,
            n_layers=n_layers,
            feed_forward_hidden=feed_forward_hidden,
            norm_config=norm_config,
            activation_config=activation_config,
            dropout_rate=dropout_rate,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
            **kwargs,
        )

        self.n_sublayers = n_sublayers
        self.agg = agg

    def _create_layer(self, layer_idx: int) -> nn.Module:
        """Creates a single GAT multi-head attention layer.

        Args:
            layer_idx: Index of the layer being created.

        Returns:
            nn.Module: GATMultiHeadAttentionLayer instance.
        """
        return GATMultiHeadAttentionLayer(
            n_heads=self.n_heads,
            embed_dim=self.embed_dim,
            feed_forward_hidden=self.feed_forward_hidden,
            norm_config=self.norm_config,
            activation_config=self.activation_config,
            connection_type=self.conn_type,
            expansion_rate=self.expansion_rate,
        )
