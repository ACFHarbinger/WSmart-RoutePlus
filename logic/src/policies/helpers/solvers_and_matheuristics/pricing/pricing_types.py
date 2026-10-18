"""Shared types of the RCSPP pricing solver.

Attributes:
    PricingStatus: Certification state of a pricing solve.

Example:
    >>> PricingStatus.EXHAUSTIVE.value
    'exhaustive'
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple


class PricingStatus(str, Enum):
    """Explicit certification state of a pricing subproblem solve."""

    EXHAUSTIVE = "exhaustive"
    HEURISTIC = "heuristic"
    PARTIAL = "partial"
    TIMED_OUT = "timed_out"


# Type alias for LCI cover items passed to the DP extension step.
# Each tuple: (cover_set, node_alpha_dict, dual_value, source_arc_or_none)
# source_arc_or_none is (i, j) for arc-saturation LCI, None for node/capacity LCI.
_LCICoverItem = Tuple[FrozenSet[int], Dict[int, float], float, Optional[Tuple[int, int]]]


# Multistar cut item: (cover_set, dual).  Capacity Q comes from self.capacity.
_MultistarItem = Tuple[FrozenSet[int], float]
