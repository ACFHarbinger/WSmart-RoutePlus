"""Unified context embedding modules for VRP variants.

This package provides context embedding layers that extract problem-specific
step features from node embeddings and environment state.

Attributes:
    CONTEXT_EMBEDDING_REGISTRY (Dict[str, Any]): Mapping of environment names
        to their respective context embedding classes.

Example:
    >>> from logic.src.models.subnets.embeddings.context import PTPContextEmbedder
    >>> embedder = PTPContextEmbedder(embed_dim=128)
"""

from __future__ import annotations

from typing import Any, Dict

from .base import ContextEmbedder
from .generic import GenericContextEmbedder
from .mvptp import MVPTPContextEmbedder
from .ptp import PTPContextEmbedder

CONTEXT_EMBEDDING_REGISTRY: Dict[str, Any] = {
    "ptp": PTPContextEmbedder,
    "mvptp": MVPTPContextEmbedder,
    "tcmvptp": MVPTPContextEmbedder,
}

__all__: list[str] = [
    "ContextEmbedder",
    "MVPTPContextEmbedder",
    "GenericContextEmbedder",
    "PTPContextEmbedder",
    "CONTEXT_EMBEDDING_REGISTRY",
]
