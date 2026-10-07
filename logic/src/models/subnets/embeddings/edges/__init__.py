"""Edge embedding modules for graph-based encoders.

This package provides edge embedding layers that project pairwise node
relationships (e.g., distances, relative positions) into the latent space.

Attributes:
    EDGE_EMBEDDING_REGISTRY (Dict[str, Any]): Mapping of environment names
        to their respective edge embedding classes.

Example:
    >>> from logic.src.models.subnets.embeddings.edges import get_edge_embedding
    >>> embedder = get_edge_embedding("vrpp", embed_dim=128)
"""

from __future__ import annotations

from typing import Any, Dict

from torch import nn

from .base import EdgeEmbedding
from .cvrpp import CVRPPEdgeEmbedding
from .none import NoEdgeEmbedding
from .tsp import TSPEdgeEmbedding

EDGE_EMBEDDING_REGISTRY: Dict[str, Any] = {
    "vrpp": TSPEdgeEmbedding,
    "cvrpp": CVRPPEdgeEmbedding,
    "ctop": CVRPPEdgeEmbedding,
    "none": NoEdgeEmbedding,
}


def get_edge_embedding(
    env_name: str,
    embed_dim: int = 128,
    **kwargs: Any,
) -> nn.Module:
    """Gets problem-specific edge embedding module.

    Args:
        env_name: Environment/problem name.
        embed_dim: Edge embedding dimension.
        kwargs: Additional arguments passed to the embedding constructor.

    Returns:
        nn.Module: Initialized edge embedding module.

    Raises:
        ValueError: If the environment name is not registered.
    """
    if env_name not in EDGE_EMBEDDING_REGISTRY:
        raise ValueError(
            f"Unknown environment for edge embedding: {env_name}. Available: {list(EDGE_EMBEDDING_REGISTRY.keys())}"
        )
    return EDGE_EMBEDDING_REGISTRY[env_name](embed_dim=embed_dim, **kwargs)


__all__: list[str] = [
    "EdgeEmbedding",
    "CVRPPEdgeEmbedding",
    "NoEdgeEmbedding",
    "TSPEdgeEmbedding",
    "EDGE_EMBEDDING_REGISTRY",
    "get_edge_embedding",
]
