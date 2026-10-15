"""TGC component factory.

Attributes:
    TGCComponentFactory: Factory for creating Transformer Graph Convolution Encoders and corresponding decoders.

Example:
    >>> from logic.src.models.subnets.factories.tgc import TGCComponentFactory
    >>> factory = TGCComponentFactory()
    >>> encoder = factory.create_encoder(embed_dim=128)
"""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.tgc import TransGraphConvEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class TGCComponentFactory(NeuralComponentFactory):
    """Factory for Transformer Graph Convolution Models.

    Attributes:
        None: Factory class does not maintain state.
    """

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create Transformer Graph Convolution Encoder.

        Args:
            kwargs: Architecture-specific configuration passed to TransGraphConvEncoder.

        Returns:
            nn.Module: The instantiated TransGraphConvEncoder.
        """
        return TransGraphConvEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type.

        Args:
            decoder_type: Type of decoder instance to create. Defaults to 'attention'.
            kwargs: Architecture-specific configuration passed to the decoder.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        return _create_decoder_by_type(decoder_type, **kwargs)
