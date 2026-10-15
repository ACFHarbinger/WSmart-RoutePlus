"""
GAC component factory.

Attributes:
    GACComponentFactory: Factory for creating Graph Attention Convolution Encoders and corresponding decoders.

Example:
    >>> from logic.src.models.subnets.factories.gac import GACComponentFactory
    >>> factory = GACComponentFactory()
    >>> encoder = factory.create_encoder(embed_dim=128, n_heads=8)
"""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.gac import GraphAttConvEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class GACComponentFactory(NeuralComponentFactory):
    """
    Factory for Graph Attention Convolution Models.

    Attributes:
        None: Factory class does not maintain state.
    """

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create Graph Attention Convolution Encoder.

        Args:
            kwargs: Architecture-specific configuration passed to GraphAttConvEncoder.

        Returns:
            nn.Module: The instantiated GraphAttConvEncoder.
        """
        return GraphAttConvEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type.

        Args:
            decoder_type: Type of decoder instance to create. Defaults to 'attention'.
            kwargs: Architecture-specific configuration passed to the decoder.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        return _create_decoder_by_type(decoder_type, **kwargs)
