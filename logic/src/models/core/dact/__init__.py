"""DACT: Dual Aspect Collaborative Transformer.

This package implements the DACT architecture (Ma et al. 2021) for iterative
improvement of routing solutions. It decomposes the problem into spatial
(node) and sequential (tour position) aspects.

Attributes:
    DACT: Training and inference wrapper.
    DACTPolicy: Neural improvement policy.
    DACTEncoder: Dual aspect Transformer encoder.
    DACTDecoder: Pairwise move prediction decoder.

"""

from .decoder import DACTDecoder as DACTDecoder
from .encoder import DACTEncoder as DACTEncoder
from .model import DACT as DACT
from .policy import DACTPolicy as DACTPolicy

__all__ = ["DACT", "DACTPolicy", "DACTEncoder", "DACTDecoder"]
