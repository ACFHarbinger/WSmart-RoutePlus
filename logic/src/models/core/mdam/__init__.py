"""MDAM: Multi-Decoder Attention Model.

This package provides MDAM for diverse solution generation. It uses a shared
encoder and multiple parallel decoders to branch out during construction.

Attributes:
    MDAM: Diverse training wrapper with competitive baseline.
    MDAMPolicy: Multi-path construction policy.
    mdam_rollout: Best-of-all rollout logic for baseline calculation.

"""

from .model import MDAM as MDAM
from .model import mdam_rollout as mdam_rollout
from .policy import MDAMPolicy as MDAMPolicy

__all__ = ["MDAM", "mdam_rollout", "MDAMPolicy"]
