"""
MDAM component factory.

Attributes:
    MDAMComponentFactory: Factory for creating MDAM Graph Attention Encoders and corresponding decoders.

Example:
    >>> from logic.src.models.subnets.factories.mdam import MDAMComponentFactory
    >>> factory = MDAMComponentFactory()
    >>> encoder = factory.create_encoder(embed_dim=128)
"""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.mdam import MDAMGraphAttentionEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class MDAMComponentFactory(NeuralComponentFactory):
    """
    Factory for MDAM Models.

    Attributes:
        None: Factory class does not maintain state.
    """

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create MDAM Graph Attention Encoder.

        Args:
            kwargs: Architecture-specific configuration passed to MDAMGraphAttentionEncoder.

        Returns:
            nn.Module: The instantiated MDAMGraphAttentionEncoder.
        """
        return MDAMGraphAttentionEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "mdam", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type.

        Args:
            decoder_type: Type of decoder instance to create. Defaults to 'mdam'.
            kwargs: Architecture-specific configuration passed to the decoder.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        return _create_decoder_by_type(decoder_type, **kwargs)
