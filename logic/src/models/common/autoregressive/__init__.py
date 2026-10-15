"""Autoregressive neural routing components.

This package provides foundation classes for constructive policies, which build
solutions sequentially. It includes base encoders, decoders, and a unified
policy class.

Attributes:
    AutoregressiveEncoder: Abstract foundation for state encoding.
    AutoregressiveDecoder: Abstract foundation for sequential decoding.
    AutoregressivePolicy: Standard constructive policy implementation.
    ConstructivePolicy: General constructive method interface.

"""

from .constructive import ConstructivePolicy as ConstructivePolicy
from .decoder import AutoregressiveDecoder as AutoregressiveDecoder
from .encoder import AutoregressiveEncoder as AutoregressiveEncoder
from .policy import AutoregressivePolicy as AutoregressivePolicy

__all__ = [
    "AutoregressiveEncoder",
    "AutoregressiveDecoder",
    "AutoregressivePolicy",
    "ConstructivePolicy",
]
