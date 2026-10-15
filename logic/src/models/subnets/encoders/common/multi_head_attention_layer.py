"""Base Multi-Head Attention Layer for Encoder Architectures.

Attributes:
    MultiHeadAttentionLayerBase: Base Multi-Head Attention Layer for Encoder Architectures.

Example:
    >>> import torch
    >>> from logic.src.models.subnets.encoders.common.multi_head_attention_layer import MultiHeadAttentionLayerBase
    >>> layer = MultiHeadAttentionLayerBase(n_heads=8, embed_dim=128, feed_forward_hidden=512)
    >>> x = torch.randn(1, 10, 128)
    >>> out = layer(x)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization
from logic.src.models.subnets.modules.connections import get_connection_module

from .feed_forward_sublayer import EncoderFeedForwardSubLayer


class MultiHeadAttentionLayerBase(nn.Module):
    """Base Multi-Head Attention Layer for Encoder Architectures.

    Implements the standard transformer encoder layer pattern:
    1. Multi-Head Attention + Residual Connection
    2. Layer Normalization
    3. Feed-Forward Network + Residual Connection
    4. Layer Normalization

    This base class is used by multiple encoder types (GAT, MoE) and supports:
    - Configurable normalization (batch, layer, instance, group)
    - Configurable activation functions
    - Skip, dense, or hyper-connections via the connection factory
    - 4D tensor handling for hyper-connections

    Subclasses can override the feed-forward component to use specialized
    variants (e.g., MoEFeedForward, ConvolutionalFFN).

    Attributes:
        att (nn.Module): Multi-head attention module wrapped with connection.
        norm1 (Normalization): First normalization layer (after attention).
        ff (nn.Module): Feed-forward module wrapped with connection.
        norm2 (Normalization): Second normalization layer (after feed-forward).
        norm_config (NormalizationConfig): Normalization configuration.
        activation_config (ActivationConfig): Activation function configuration.
    """

    def __init__(
        self,
        n_heads: int,
        embed_dim: int,
        feed_forward_hidden: int,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        connection_type: str = "skip",
        expansion_rate: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize the MultiHeadAttentionLayerBase.

        Args:
            n_heads: Number of attention heads.
            embed_dim: Embedding dimension (must be divisible by n_heads).
            feed_forward_hidden: Hidden dimension for feed-forward network.
            norm_config: Normalization configuration. Defaults to None.
            activation_config: Activation configuration. Defaults to None.
            connection_type: Type of connection: "skip", "dense", or "hyper". Defaults to "skip".
            expansion_rate: Expansion factor for hyper-connections. Defaults to 4.
            kwargs: Additional keyword arguments.
        """
        super(MultiHeadAttentionLayerBase, self).__init__()

        # Use default configs if none provided
        if norm_config is None:
            norm_config = NormalizationConfig()

        if activation_config is None:
            activation_config = ActivationConfig()

        # Store configs for subclass access
        self.norm_config = norm_config
        self.activation_config = activation_config

        # 1. Multi-Head Attention with connection wrapper
        self.att = get_connection_module(
            module=MultiHeadAttention(
                n_heads=n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim,
            ),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

        # 2. First normalization (after attention)
        self.norm1 = Normalization(
            embed_dim,
            norm_config=norm_config,
        )

        # 3. Feed-Forward Network with connection wrapper
        self.ff = self._create_feed_forward(
            embed_dim=embed_dim,
            feed_forward_hidden=feed_forward_hidden,
            activation_config=activation_config,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

        # 4. Second normalization (after feed-forward)
        self.norm2 = Normalization(
            embed_dim,
            norm_config=norm_config,
        )

    def _create_feed_forward(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation_config: ActivationConfig,
        connection_type: str,
        expansion_rate: int,
    ) -> nn.Module:
        """Create the feed-forward component.

        Subclasses can override this to use specialized feed-forward variants
        (e.g., MoEFeedForward, ConvolutionalFFN).

        Args:
            embed_dim: Embedding dimension.
            feed_forward_hidden: Hidden dimension for feed-forward network.
            activation_config: Activation function configuration.
            connection_type: Type of connection wrapper.
            expansion_rate: Expansion factor for hyper-connections.

        Returns:
            nn.Module: Feed-forward module wrapped with connection.
        """
        return get_connection_module(
            module=EncoderFeedForwardSubLayer(
                embed_dim,
                feed_forward_hidden,
                activation_config=activation_config,
            ),
            embed_dim=embed_dim,
            connection_type=connection_type,
            expansion_rate=expansion_rate,
        )

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the attention layer.

        Args:
            h: Input tensor of shape (batch, seq_len, embed_dim)
                or (batch, seq_len, embed_dim, n) for hyper-connections.
            mask: Attention mask of shape (batch, seq_len)
                or (batch, seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        # 1. Multi-Head Attention
        h = self.att(h, mask=mask)

        # 2. First normalization (handle 4D for hyper-connections)
        h = self._apply_norm(h, self.norm1)

        # 3. Feed-Forward Network
        h = self.ff(h)

        # 4. Second normalization (handle 4D for hyper-connections)
        h = self._apply_norm(h, self.norm2)

        return h

    def _apply_norm(self, h: torch.Tensor, norm_layer: nn.Module) -> torch.Tensor:
        """Apply normalization, handling both 3D and 4D tensors.

        For hyper-connections, the input is 4D: (batch, seq_len, embed_dim, n).
        We need to permute to (batch, seq_len, n, embed_dim) for normalization,
        then permute back.

        Args:
            h: Input tensor (3D or 4D).
            norm_layer: Normalization layer to apply.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        if h.dim() == 4:  # Hyper-connection: (batch, seq_len, embed_dim, n)
            # Permute to (batch, seq_len, n, embed_dim) for Norm(embed_dim)
            h_perm = h.permute(0, 1, 3, 2).contiguous()
            h_norm: torch.Tensor = norm_layer(h_perm)
            # Permute back to (batch, seq_len, embed_dim, n)
            return h_norm.permute(0, 1, 3, 2)
        else:  # Standard: (batch, seq_len, embed_dim)
            return norm_layer(h)
