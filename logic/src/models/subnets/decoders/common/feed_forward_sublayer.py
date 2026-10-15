"""Base decoder classes for reducing boilerplate across decoder implementations.

This module provides reusable components for decoder architectures, specifically
focusing on standard feed-forward network (FFN) patterns.

Attributes:
    FeedForwardSubLayer: Shared feed-forward sublayer for decoders.

Example:
    >>> from logic.src.models.subnets.decoders.common.feed_forward_sublayer import FeedForwardSubLayer
    >>> ff_layer = FeedForwardSubLayer(embed_dim=128, feed_forward_hidden=512)
    >>> output = ff_layer(input_tensor)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.models.subnets.modules import ActivationFunction, FeedForward


class FeedForwardSubLayer(nn.Module):
    """Reusable feed-forward sub-layer for decoder architectures.

    Implements the standard transformer feed-forward pattern:
        FFN(x) = Activation(W1 * x + b1) * W2 + b2

    This layer is used in GAT decoders, Deep decoders, and other attention-based
    decoder architectures.

    Attributes:
        sub_layers (nn.Sequential): Sequential container with FFN layers and activation.
    """

    def __init__(
        self,
        embed_dim: int,
        feed_forward_hidden: int,
        activation_config: Optional[ActivationConfig] = None,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initializes the FeedForwardSubLayer.

        Args:
            embed_dim: Input and output embedding dimension.
            feed_forward_hidden: Hidden dimension for the feed-forward network.
            activation_config: Activation function configuration.
            bias: Whether to use bias in linear layers.
            kwargs: Additional keyword arguments.
        """
        super(FeedForwardSubLayer, self).__init__()

        # Default config if not provided
        if activation_config is None:
            activation_config = ActivationConfig()

        # Standard FFN: Linear -> Activation -> Linear
        if feed_forward_hidden > 0:
            self.sub_layers = nn.Sequential(
                # Up-projection: embed_dim -> feed_forward_hidden
                FeedForward(embed_dim, feed_forward_hidden, bias=bias),
                # Activation function
                ActivationFunction(
                    activation_config.name,
                    activation_config.param,
                    activation_config.threshold,
                    activation_config.replacement_value,
                    activation_config.n_params,
                    tuple(activation_config.range) if activation_config.range else None,  # type: ignore[arg-type]
                ),
                # Down-projection: feed_forward_hidden -> embed_dim
                FeedForward(feed_forward_hidden, embed_dim, bias=bias),
            )
        else:
            # Degenerate case: single linear layer (no activation)
            self.sub_layers = FeedForward(embed_dim, embed_dim, bias=bias)  # type: ignore[assignment]

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through feed-forward sublayer.

        Args:
            h: Input tensor of shape (batch_size, seq_len, embed_dim).
            mask: Mask tensor (currently unused).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        return self.sub_layers(h)
