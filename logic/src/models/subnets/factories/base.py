"""
Abstract base for neural component factories and decoder utility.

Attributes:
    NeuralComponentFactory: Abstract interface for creating encoders and decoders.
    _create_decoder_by_type: Internal utility for instantiating decoders by string identifiers.

Example:
    >>> # Factories are usually inherited or used via specific implementations
    >>> from logic.src.models.subnets.factories.base import NeuralComponentFactory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.utils.registry import import_target

_DECODER_TARGETS = {
    "attention": "logic.src.models.subnets.decoders.glimpse.decoder:GlimpseDecoder",
    "glimpse": "logic.src.models.subnets.decoders.glimpse.decoder:GlimpseDecoder",
    "deep": "logic.src.models.subnets.decoders.gat:DeepGATDecoder",
    "gat": "logic.src.models.subnets.decoders.gat:DeepGATDecoder",
    "graph_attention": "logic.src.models.subnets.decoders.gat:DeepGATDecoder",
    "pointer": "logic.src.models.subnets.decoders.ptr:PointerDecoder",
    "mdam": "logic.src.models.subnets.decoders.mdam:MDAMDecoder",
    "polynet": "logic.src.models.subnets.decoders.polynet:PolyNetDecoder",
    "aco": "logic.src.models.subnets.decoders.deepaco:ACODecoder",
}


def _create_decoder_by_type(decoder_type: str, **kwargs: Any) -> nn.Module:
    """Instantiates a decoder based on the specified type name."""
    decoder_type = decoder_type.lower()
    target = _DECODER_TARGETS.get(decoder_type)
    if target is None:
        raise ValueError(
            f"Unknown decoder_type: {decoder_type}. Choose from 'attention', 'glimpse', 'deep', 'pointer', 'mdam', 'polynet', 'aco'."
        )
    decoder_cls = import_target(target)
    return decoder_cls(**kwargs)


class NeuralComponentFactory(ABC):
    """
    Abstract Factory for creating neural components (Encoders and Decoders).

    Attributes:
        None: Abstract base class does not maintain state.
    """

    @abstractmethod
    def create_encoder(
        self,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create an encoder instance.

        Args:
            norm_config (Optional[NormalizationConfig]): Normalization settings. Defaults to None.
            activation_config (Optional[ActivationConfig]): Activation settings. Defaults to None.
            kwargs: Architecture-specific configuration.

        Returns:
            nn.Module: The instantiated encoder module.
        """
        pass

    @abstractmethod
    def create_decoder(
        self,
        decoder_type: str = "attention",
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create a decoder instance based on decoder_type.

        Args:
            decoder_type (str): Type of decoder ('attention', 'deep', 'pointer'). Defaults to 'attention'.
            norm_config (Optional[NormalizationConfig]): Normalization settings. Defaults to None.
            activation_config (Optional[ActivationConfig]): Activation settings. Defaults to None.
            kwargs: Architecture-specific configuration.

        Returns:
            nn.Module: The instantiated decoder module.
        """
        pass
