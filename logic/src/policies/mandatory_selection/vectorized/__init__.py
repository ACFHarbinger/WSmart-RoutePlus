"""Selection strategies package.

This package provides vectorized bin selection strategies for identifying
mandatory nodes across a batch of environments. It includes reactive,
periodic, predictive, and neural network-based heuristics.

Attributes:
    VectorizedSelector: Abstract base class for all selectors.
    LastMinuteSelector: Threshold-based reactive selection.
    RegularSelector: Periodic collection on scheduled days.
    LookaheadSelector: Predictive overflow-based selection.
    RevenueSelector: Revenue-based selection.
    ServiceLevelSelector: Statistical overflow prediction.
    CombinedSelector: Logic-based combination of multiple strategies.
    ManagerSelector: Neural network-based selection (MandatoryManager).
    get_vectorized_selector: Factory function for easy instantiation by name.
    create_selector_from_config: Factory for configuration-driven instantiation.

Example:
    None
"""

from __future__ import annotations

from .base import VectorizedSelector
from .combined import CombinedSelector
from .factory import create_selector_from_config, get_vectorized_selector
from .last_minute import LastMinuteSelector
from .lookahead import LookaheadSelector
from .manager import ManagerSelector
from .regular import RegularSelector
from .revenue import RevenueSelector
from .service_level import ServiceLevelSelector

__all__ = [
    "VectorizedSelector",
    "LastMinuteSelector",
    "RegularSelector",
    "LookaheadSelector",
    "RevenueSelector",
    "ServiceLevelSelector",
    "CombinedSelector",
    "ManagerSelector",
    "get_vectorized_selector",
    "create_selector_from_config",
]
