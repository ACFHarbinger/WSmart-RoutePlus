"""Temporal Attention Model (TAM) components.

This package provides the implementation of the Temporal Attention Model,
which extends graph attention with recurrent forecasting for dynamic routing.

Attributes:
    TemporalAttentionModel: Base AM variant with history-aware node encoding.
    TemporalAMPolicy: RL4CO-compatible wrapper with proactive state estimation.

"""

from .model import TemporalAttentionModel as TemporalAttentionModel
from .policy import TemporalAMPolicy as TemporalAMPolicy

__all__ = ["TemporalAttentionModel", "TemporalAMPolicy"]
