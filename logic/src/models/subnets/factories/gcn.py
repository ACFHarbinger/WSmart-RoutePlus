"""
GCN component factory.

Attributes:
    GCNComponentFactory: Factory for creating Graph Convolutional Network Encoders and corresponding decoders.

Example:
    >>> from logic.src.models.subnets.factories.gcn import GCNComponentFactory
    >>> factory = GCNComponentFactory()
    >>> encoder = factory.create_encoder(embed_dim=128, n_layers=3)
"""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.gcn import GraphConvolutionEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class GCNComponentFactory(NeuralComponentFactory):
    """
    Factory for Graph Convolutional Network Models.

    Attributes:
        None: Factory class does not maintain state.
    """

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create GCN Encoder.

        Args:
            kwargs: Architecture-specific configuration passed to GraphConvolutionEncoder.

        Returns:
            nn.Module: The instantiated GraphConvolutionEncoder.
        """
        return GraphConvolutionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type.

        Args:
            decoder_type: Type of decoder instance to create. Defaults to 'attention'.
            kwargs: Architecture-specific configuration passed to the decoder.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        return _create_decoder_by_type(decoder_type, **kwargs)
