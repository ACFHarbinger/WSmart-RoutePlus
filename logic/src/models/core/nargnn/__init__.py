"""NARGNN: Non-Autoregressive Graph Neural Networks.

This package provides non-autoregressive models that use GNNs to predict
edge discovery heatmaps for efficient construction.

Attributes:
    NARGNN: REINFORCE training wrapper for heatmap models.
    NARGNNPolicy: Heatmap prediction and decoding policy.

"""

from .model import NARGNN as NARGNN
from .policy import NARGNNPolicy as NARGNNPolicy

__all__ = ["NARGNN", "NARGNNPolicy"]
