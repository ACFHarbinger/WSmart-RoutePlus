"""Unstringing operators package.

This package provides GPU-accelerated implementations of the sophisticated
unstringing operators family (Types I-IV). Unstringing operators improve
tour quality by removing a node and reconnecting the route with multiple
segment reversals, creating powerful moves that can escape deep local optima.

Attributes:
    vectorized_type_i_unstringing: Type I operator (2 segment reversals).
    vectorized_type_ii_unstringing: Type II operator (alternative reversal ordering).
    vectorized_type_iii_unstringing: Type III operator (3 segment reversals).
    vectorized_type_iv_unstringing: Type IV operator (selective reversals).

Example:
    >>> from logic.src.policies.vectorized.routing.operators import unstringing
    >>> opt_tours = unstringing.vectorized_type_i_unstringing(tours, dist)
"""

from __future__ import annotations

from .type_i import vectorized_type_i_unstringing
from .type_ii import vectorized_type_ii_unstringing
from .type_iii import vectorized_type_iii_unstringing
from .type_iv import vectorized_type_iv_unstringing

__all__ = [
    "vectorized_type_i_unstringing",
    "vectorized_type_ii_unstringing",
    "vectorized_type_iii_unstringing",
    "vectorized_type_iv_unstringing",
]
