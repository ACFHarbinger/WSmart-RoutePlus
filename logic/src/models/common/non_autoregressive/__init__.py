"""Non-autoregressive (heatmap-based) neural routing components.

This package provides foundation classes for policies that predict global
graph features (heatmaps) to guide one-shot solution construction.

Attributes:
    NonAutoregressiveEncoder: Abstract foundation for heatmap prediction.
    NonAutoregressiveDecoder: Abstract foundation for construction from heatmaps.
    NonAutoregressivePolicy: Standard non-autoregressive policy implementation.

"""

from .decoder import NonAutoregressiveDecoder as NonAutoregressiveDecoder
from .encoder import NonAutoregressiveEncoder as NonAutoregressiveEncoder
from .policy import NonAutoregressivePolicy as NonAutoregressivePolicy

__all__ = [
    "NonAutoregressiveEncoder",
    "NonAutoregressiveDecoder",
    "NonAutoregressivePolicy",
]
