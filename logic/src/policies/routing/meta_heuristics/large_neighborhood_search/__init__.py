r"""Shared large-neighborhood-search engine (WP-11 P-31).

One LNS loop skeleton serves the released classic ALNS and the PG-CLNS
coaching LNS. Per-class semantics (scoring, visited-set tracking, copy
policy, removal bounds, acceptance, weight update) are injected through
small hooks on :class:`LNSBase`.

Attributes:
    LNSBase: Shared LNS loop skeleton.
    MIN_REMOVAL_LOWER_BOUND: Named lower bound for the classic ALNS removal count.
    RopkePisingerSegments: Segment-based weight update (classic ALNS).
    EmaWeightUpdate: Per-iteration exponential moving-average weight update (PG-CLNS).

"""

from .engine import MIN_REMOVAL_LOWER_BOUND, LNSBase
from .weight_updates import EmaWeightUpdate, RopkePisingerSegments

__all__ = [
    "LNSBase",
    "MIN_REMOVAL_LOWER_BOUND",
    "RopkePisingerSegments",
    "EmaWeightUpdate",
]
