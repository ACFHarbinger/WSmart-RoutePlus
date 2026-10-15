"""Improvement-based neural search components.

This package provides classes for improvement policies, which refine existing
solutions through iterative local search or optimization operators.

Attributes:
    ImprovementEncoder: Foundation for solution-aware state encoding.
    ImprovementDecoder: Foundation for predicting improvement moves.
    ImprovementPolicy: Standard iterative refinement policy implementation.

"""

from .decoder import ImprovementDecoder as ImprovementDecoder
from .encoder import ImprovementEncoder as ImprovementEncoder
from .policy import ImprovementPolicy as ImprovementPolicy

__all__ = [
    "ImprovementEncoder",
    "ImprovementDecoder",
    "ImprovementPolicy",
]
