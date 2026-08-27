"""Unified context embedding modules for VRP variants.

This package provides context embedding layers that extract problem-specific
step features from node embeddings and environment state.

Attributes:
    CONTEXT_EMBEDDING_REGISTRY (Dict[str, Any]): Mapping of environment names
        to their respective context embedding classes.

Example:
    >>> from logic.src.models.subnets.embeddings.context import VRPPContextEmbedder
    >>> embedder = VRPPContextEmbedder(embed_dim=128)
"""

from __future__ import annotations

from typing import Any, Dict

from .base import ContextEmbedder
from .cvrpp import CVRPPContextEmbedder
from .generic import GenericContextEmbedder
from .vrpp import VRPPContextEmbedder
from .wcvrp import WCVRPContextEmbedder

CONTEXT_EMBEDDING_REGISTRY: Dict[str, Any] = {
    "vrpp": VRPPContextEmbedder,
    "cvrpp": CVRPPContextEmbedder,
    "ttop": CVRPPContextEmbedder,
    "wcvrp": WCVRPContextEmbedder,
    "cwcvrp": WCVRPContextEmbedder,
    "sdwcvrp": WCVRPContextEmbedder,
    "swcvrp": WCVRPContextEmbedder,
    "scwcvrp": WCVRPContextEmbedder,
}

__all__: list[str] = [
    "ContextEmbedder",
    "CVRPPContextEmbedder",
    "GenericContextEmbedder",
    "VRPPContextEmbedder",
    "WCVRPContextEmbedder",
    "CONTEXT_EMBEDDING_REGISTRY",
]
