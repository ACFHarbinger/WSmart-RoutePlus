"""Neural Optimizer (NeuOpt).

This package implements the NeuOpt architecture for iterative combinatorial
optimization. It provides a Transformer-based global state encoder and
a pairwise decoder for identifying high-quality local improvements.

Attributes:
    NeuOpt: Primary solver wrapper for neural iterative optimization.
    NeuOptPolicy: Policy coordinating NeuOpt encoder and decoder.
    NeuOptEncoder: Deep Transformer node-feature representation encoder.
    NeuOptDecoder: Pairwise action decoder for move selection.

"""

from .decoder import NeuOptDecoder as NeuOptDecoder
from .encoder import NeuOptEncoder as NeuOptEncoder
from .model import NeuOpt as NeuOpt
from .policy import NeuOptPolicy as NeuOptPolicy

__all__ = ["NeuOpt", "NeuOptPolicy", "NeuOptEncoder", "NeuOptDecoder"]
