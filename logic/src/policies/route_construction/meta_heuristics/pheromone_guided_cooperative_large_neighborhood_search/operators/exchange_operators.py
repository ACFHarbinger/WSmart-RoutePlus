"""
Exchange operators for VRP local search.

This module contains advanced inter-route and intra-route operators:
- Or-opt: Relocate chains of consecutive nodes
- Cross-Exchange: Swap segments between routes (λ-interchange)
- Ejection Chain: Compound displacement for fleet reduction

(Refactored to point to `logic.src.policies.operators.exchange` package)

Attributes:
    or_opt_operators: Set of or-opt operators.

Example:
    >>> from logic.src.policies.operators.exchange import move_or_opt
    >>> new_routes = move_or_opt(routes, dist_matrix)
"""

from .exchange import (
    move_or_opt,
)

__all__ = [
    "move_or_opt",
]
