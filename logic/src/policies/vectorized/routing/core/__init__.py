"""Core genetic components of the vectorised experts.

Attributes:
    vectorized_ordered_crossover: Batched ordered crossover (OX1).

Example:
    >>> from logic.src.policies.vectorized.routing.core import vectorized_ordered_crossover
"""

from __future__ import annotations

from .crossover import vectorized_ordered_crossover

__all__ = ["vectorized_ordered_crossover"]
