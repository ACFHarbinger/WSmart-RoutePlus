"""State embedding modules for environment representations.

This package provides classes that encapsulate environment states (e.g., VRPP,
WCVRP) for consistent processing by RL models and context embedders.

Attributes:
    STATE_EMBEDDING_REGISTRY (Dict[str, Any]): Mapping of environment names
        to their respective state classes.

Example:
    >>> from logic.src.models.subnets.embeddings.state import STATE_EMBEDDING_REGISTRY
    >>> state_cls = STATE_EMBEDDING_REGISTRY["vrpp"]
"""

from __future__ import annotations

from typing import Any, Dict

from .cvrpp import CVRPPState
from .env import EnvState
from .swcvrp import SWCVRPState
from .vrpp import VRPPState
from .wcvrp import WCVRPState

STATE_EMBEDDING_REGISTRY: Dict[str, Any] = {
    "vrpp": VRPPState,
    "cvrpp": CVRPPState,
    "ttop": CVRPPState,
    "wcvrp": WCVRPState,
    "cwcvrp": WCVRPState,
    "sdwcvrp": WCVRPState,
    "swcvrp": SWCVRPState,
    "scwcvrp": SWCVRPState,
}

__all__: list[str] = [
    "EnvState",
    "VRPPState",
    "WCVRPState",
    "CVRPPState",
    "SWCVRPState",
    "STATE_EMBEDDING_REGISTRY",
]
