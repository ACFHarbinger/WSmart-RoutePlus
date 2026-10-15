"""NAR component factory.

Attributes:
    NARComponentFactory: Factory for creating Non-Autoregressive Encoders (NARGNN) and corresponding decoders.

Example:
    >>> from logic.src.models.subnets.factories.nar import NARComponentFactory
    >>> factory = NARComponentFactory()
    >>> encoder = factory.create_encoder(embed_dim=128)
"""

from __future__ import annotations

from typing import Any

from torch import nn

from logic.src.models.subnets.encoders.nargnn import NARGNNEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class NARComponentFactory(NeuralComponentFactory):
    """Factory for Non-Autoregressive Models (DeepACO, GFACS, NARGNN).

    Attributes:
        None: Factory class does not maintain state.
    """

    def create_encoder(self, **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create NARGNN Encoder.

        Args:
            kwargs: Architecture-specific configuration passed to NARGNNEncoder.

        Returns:
            nn.Module: The instantiated NARGNNEncoder.
        """
        return NARGNNEncoder(**kwargs)

    def create_decoder(self, decoder_type: str = "aco", **kwargs: Any) -> nn.Module:  # type: ignore[override]
        """Create decoder based on decoder_type.

        Args:
            decoder_type: Type of decoder instance to create. Defaults to 'aco'.
            kwargs: Architecture-specific configuration passed to the decoder.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        return _create_decoder_by_type(decoder_type, **kwargs)
