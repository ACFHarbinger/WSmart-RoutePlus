"""State embedding modules for environment representations.

This package provides classes that encapsulate environment states (e.g., PTP,
CPTP) for consistent processing by RL models and context embedders.

Attributes:
    STATE_EMBEDDING_REGISTRY (Dict[str, Any]): Mapping of environment names
        to their respective state classes.

Example:
    >>> from logic.src.models.subnets.embeddings.state import STATE_EMBEDDING_REGISTRY
    >>> state_cls = STATE_EMBEDDING_REGISTRY["ptp"]
"""

from __future__ import annotations

from typing import Any, Dict

from .env import EnvState
from .cptp import CPTPState
from .ptp import PTPState

STATE_EMBEDDING_REGISTRY: Dict[str, Any] = {
    "ptp": PTPState,
    "cptp": CPTPState,
    "tcptp": CPTPState,
}

__all__: list[str] = [
    "EnvState",
    "PTPState",
    "CPTPState",
    "STATE_EMBEDDING_REGISTRY",
]
