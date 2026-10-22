"""Route-level optimization operators.

This package contains vectorized local search operators that focus on
optimizing single sequences or inter-route connectivity without significantly
altering the node visit set across the entire fleet.

Attributes:
    vectorized_two_opt: 2-opt local search (edge-swap segment reversal).
    vectorized_three_opt: 3-opt local search (three-edge reconnection).
    vectorized_two_opt_star: 2-opt* inter-route tail swap.
    vectorized_swap_star: Inter-route node exchange with optimal re-insertion.
    vectorized_lkh: Lin-Kernighan-Helsgaun sophisticated local search.

Example:
    None
"""

from __future__ import annotations

from .lkh import vectorized_lkh
from .swap_star import vectorized_swap_star
from .three_opt import vectorized_three_opt
from .two_opt import vectorized_two_opt
from .two_opt_star import vectorized_two_opt_star

__all__ = [
    "vectorized_two_opt",
    "vectorized_three_opt",
    "vectorized_two_opt_star",
    "vectorized_swap_star",
    "vectorized_lkh",
]
