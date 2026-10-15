"""Pointer Network architectures and policies.

This package provides the classic sequence-to-sequence pointer mechanism for
combinatorial optimization, implemented using recurrent neural networks.

Attributes:
    PointerNetwork: Standalone LSTM-based pointer architecture.
    PointerNetworkPolicy: Policy adapter for standard training loops.

"""

from .model import PointerNetwork as PointerNetwork
from .policy import PointerNetworkPolicy as PointerNetworkPolicy

__all__ = ["PointerNetwork", "PointerNetworkPolicy"]
