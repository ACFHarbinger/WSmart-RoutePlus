"""Neural Neighborhood Search (N2S).

This package implements the N2S architecture for iterative solution refinement.
It provides a specialized encoder with neighborhood masking and a pairwise
decoder for selecting local moves.

Attributes:
    N2S: Unified training and inference wrapper for neighborhood search.
    N2SPolicy: Policy coordinating N2S encoder and decoder.
    N2SEncoder: Sparse Transformer encoder with spatial neighborhood masking.
    N2SDecoder: Pairwise action decoder for move selection.

"""

from .decoder import N2SDecoder as N2SDecoder
from .encoder import N2SEncoder as N2SEncoder
from .model import N2S as N2S
from .policy import N2SPolicy as N2SPolicy

__all__ = ["N2S", "N2SPolicy", "N2SEncoder", "N2SDecoder"]
