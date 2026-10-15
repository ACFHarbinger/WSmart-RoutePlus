"""
MLP component factory.

Attributes:
    MLPComponentFactory: Factory for creating Multi-Layer Perceptron Encoders and corresponding decoders.

Example:
    >>> from logic.src.models.subnets.factories.mlp import MLPComponentFactory
    >>> factory = MLPComponentFactory()
    >>> encoder = factory.create_encoder(embed_dim=128, n_layers=3)
"""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.mlp import MLPEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class MLPComponentFactory(NeuralComponentFactory):
    """
    Factory for MLP-based Models.

    Attributes:
        None: Factory class does not maintain state.
    """

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create MLP Encoder.

        Args:
            kwargs: Architecture-specific configuration passed to MLPEncoder.

        Returns:
            nn.Module: The instantiated MLPEncoder.
        """
        return MLPEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "attention", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type.

        Args:
            decoder_type: Type of decoder instance to create. Defaults to 'attention'.
            kwargs: Architecture-specific configuration passed to the decoder.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        return _create_decoder_by_type(decoder_type, **kwargs)
