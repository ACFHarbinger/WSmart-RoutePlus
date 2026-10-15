"""Attention Model-based routing components.

This package implements the Attention Model (AM) and its variants for
solving combinatorial optimization problems through constructive search.

Attributes:
    AttentionModel: The base encoder-decoder implementation.
    AttentionModelPolicy: RL4CO-compatible wrapper for training.
    DeepDecoderPolicy: Version with a multi-layer attention decoder.
    SymNCOPolicy: Invariant policy with a projection head.

"""

from .deep_decoder_policy import DeepDecoderPolicy as DeepDecoderPolicy
from .model import AttentionModel as AttentionModel
from .policy import AttentionModelPolicy as AttentionModelPolicy
from .symnco_policy import SymNCOPolicy as SymNCOPolicy

__all__ = [
    "AttentionModel",
    "AttentionModelPolicy",
    "DeepDecoderPolicy",
    "SymNCOPolicy",
]
