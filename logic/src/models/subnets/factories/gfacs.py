"""
GFACS component factory.

Attributes:
    GFACSComponentFactory: Factory for creating GFACS Encoders and corresponding decoders.

Example:
    >>> from logic.src.models.subnets.factories.gfacs import GFACSComponentFactory
    >>> factory = GFACSComponentFactory()
    >>> encoder = factory.create_encoder(embed_dim=128)
"""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.decoders.deepaco import ACODecoder
from logic.src.models.subnets.encoders.gfacs.encoder import GFACSEncoder

from .base import NeuralComponentFactory


class GFACSComponentFactory(NeuralComponentFactory):
    """
    Factory for GFACS Models.

    Attributes:
        None: Factory class does not maintain state.
    """

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create GFACS Encoder.

        Args:
            kwargs: Architecture-specific configuration passed to GFACSEncoder.

        Returns:
            nn.Module: The instantiated GFACSEncoder.
        """
        return GFACSEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "aco", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create ACO Decoder.

        Args:
            decoder_type: Type of decoder instance to create. Defaults to 'aco'.
            kwargs: Architecture-specific configuration passed to ACODecoder.

        Returns:
            nn.Module: The instantiated ACODecoder module.
        """
        # Reuse ACODecoder from DeepACO
        return ACODecoder(**kwargs)
