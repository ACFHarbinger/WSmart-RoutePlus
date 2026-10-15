"""
GGAC component factory.

Attributes:
    GGACComponentFactory: Factory for creating Gated Graph Attention Convolution Encoders and corresponding decoders.

Example:
    >>> from logic.src.models.subnets.factories.ggac import GGACComponentFactory
    >>> factory = GGACComponentFactory()
    >>> encoder = factory.create_encoder(embed_dim=128)
"""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.ggac import GatedGraphAttConvEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class GGACComponentFactory(NeuralComponentFactory):
    """
    Factory for Gated Graph Attention Convolution Models.

    Attributes:
        None: Factory class does not maintain state.
    """

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create Gated Graph Attention Convolution Encoder.

        Args:
            kwargs: Architecture-specific configuration passed to GatedGraphAttConvEncoder.

        Returns:
            nn.Module: The instantiated GatedGraphAttConvEncoder.
        """
        return GatedGraphAttConvEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type.

        Args:
            decoder_type: Type of decoder instance to create. Defaults to 'attention'.
            kwargs: Architecture-specific configuration passed to the decoder.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        return _create_decoder_by_type(decoder_type, **kwargs)
