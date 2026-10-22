"""Exchange operators package.

This package provides vectorized inter-route exchange operators, including
the complex ejection chain heuristic and the generalized lambda-interchange
family of moves.

Attributes:
    vectorized_cross_exchange: Inter-route segment exchange.
    vectorized_ejection_chain: Multi-route node redistribution for fleet minimization.
    vectorized_lambda_interchange: Generalized segment exchange search (0..λ).
    vectorized_or_opt: Segment relocation within or between routes.

Example:
    None
"""

from __future__ import annotations

from .cross_exchange import vectorized_cross_exchange
from .ejection_chain import vectorized_ejection_chain
from .lambda_interchange import vectorized_lambda_interchange
from .or_opt import vectorized_or_opt

__all__ = [
    "vectorized_cross_exchange",
    "vectorized_ejection_chain",
    "vectorized_lambda_interchange",
    "vectorized_or_opt",
]
