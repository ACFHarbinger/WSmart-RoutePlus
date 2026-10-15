"""PolyNet: Learning Diverse Solution Strategies.

This package provides PolyNet models that learn multiple distinct search
behaviors for combinatorial optimization using strategy conditioning and
the Poppy loss function.

Attributes:
    PolyNet: Training wrapper for multi-strategy learning.
    PolyNetPolicy: Behavior-conditioned construction policy.

"""

from .model import PolyNet as PolyNet
from .policy import PolyNetPolicy as PolyNetPolicy

__all__ = ["PolyNet", "PolyNetPolicy"]
