"""Transductive (Test-time Adaptation) components.

This package provides classes for transductive inference, where the model
parameters are adapted to specific test instances through optimization.

Attributes:
    TransductiveModel: Base foundation for all transductive methods.
    ActiveSearch: Full model parameter optimization at test-time.
    EAS: Efficient parameter adaptation using selective subsets.
    EASEmb: EAS variant targeting state embeddings.
    EASLay: EAS variant targeting architectural layers.

"""

from .active_search import ActiveSearch as ActiveSearch
from .base import TransductiveModel as TransductiveModel
from .eas import EAS as EAS
from .eas_embeddings import EASEmb as EASEmb
from .eas_layers import EASLay as EASLay

__all__ = [
    "TransductiveModel",
    "ActiveSearch",
    "EAS",
    "EASEmb",
    "EASLay",
]
