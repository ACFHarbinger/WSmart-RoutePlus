"""
Attention model component factory.

Attributes:
    AttentionComponentFactory: Factory for creating Graph Attention Encoders and corresponding decoders.

Example:
    >>> from logic.src.models.subnets.factories.attention import AttentionComponentFactory
    >>> factory = AttentionComponentFactory()
    >>> encoder = factory.create_encoder(embed_dim=128, n_heads=8)
"""

from __future__ import annotations

from typing import Any, Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.models.subnets.encoders.gat import GraphAttentionEncoder

from .base import NeuralComponentFactory, _create_decoder_by_type


class AttentionComponentFactory(NeuralComponentFactory):
    """
    Factory for Angle-based Attention Models.

    Attributes:
        None: Factory class does not maintain state.
    """

    def create_encoder(
        self,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create Graph Attention Encoder.

        Args:
            norm_config: Normalization settings. Defaults to None.
            activation_config: Activation settings. Defaults to None.
            kwargs: Architecture-specific configuration passed to GraphAttentionEncoder.

        Returns:
            nn.Module: The instantiated GraphAttentionEncoder.
        """
        return GraphAttentionEncoder(
            norm_config=norm_config,
            activation_config=activation_config,
            **kwargs,
        )

    def create_decoder(
        self,
        decoder_type: str = "attention",
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create decoder based on decoder_type.

        Args:
            decoder_type: Type of decoder instance to create. Defaults to 'attention'.
            norm_config: Normalization settings. Defaults to None.
            activation_config: Activation settings. Defaults to None.
            kwargs: Architecture-specific configuration passed to the decoder.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        return _create_decoder_by_type(
            decoder_type,
            norm_config=norm_config,
            activation_config=activation_config,
            **kwargs,
        )
