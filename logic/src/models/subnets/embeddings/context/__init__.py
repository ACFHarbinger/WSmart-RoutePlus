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

CONTEXT_EMBEDDING_REGISTRY: Dict[str, Any] = {
    "vrpp": VRPPContextEmbedder,
    "cvrpp": CVRPPContextEmbedder,
    "ctop": CVRPPContextEmbedder,
}

__all__: list[str] = [
    "ContextEmbedder",
    "CVRPPContextEmbedder",
    "GenericContextEmbedder",
    "VRPPContextEmbedder",
    "CONTEXT_EMBEDDING_REGISTRY",
]
