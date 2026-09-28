"""Unified cache for attention-based decoder architectures.

This module provides a unified cache structure that consolidates multiple
decoder-specific cache implementations into a single, high-performance object.

Attributes:
    AttentionDecoderCache: Unified cache for attention-based decoders (Glimpse, DeepGAT, etc.).

Example:
    >>> from logic.src.models.subnets.decoders.common.cache import AttentionDecoderCache
    >>> cache = AttentionDecoderCache(node_embeddings=emb, graph_context=ctx)
    >>> sub_cache = cache[0:10]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class AttentionDecoderCache:
    """Unified cache for attention-based decoder precomputations.

    This cache stores precomputed projections of encoder outputs that remain
    fixed during autoregressive decoding. Using a cache avoids recomputing
    these projections at each decoding step, significantly improving performance.

    The cache supports different decoder architectures:
    - Full multi-head attention decoders (Glimpse, PolyNet, MDAM): Use all fields.
    - Simplified decoders (GAT): Use only node_embeddings + graph_context.

    Attributes:
        node_embeddings (torch.Tensor): Original node embeddings from encoder.
            Shape: (batch_size, num_nodes, embed_dim).
        graph_context (torch.Tensor): Projected graph-level context.
            Shape: (batch_size, 1, embed_dim) or (batch_size, embed_dim).
        glimpse_key (Optional[torch.Tensor]): Precomputed keys for glimpse attention.
            Shape: (batch_size, num_heads, num_nodes, key_dim).
        glimpse_val (Optional[torch.Tensor]): Precomputed values for glimpse attention.
            Shape: (batch_size, num_heads, num_nodes, val_dim).
        logit_key (Optional[torch.Tensor]): Precomputed keys for final logit computation.
            Shape: (batch_size, num_nodes, embed_dim).
    """

    node_embeddings: torch.Tensor
    graph_context: Optional[torch.Tensor] = None
    glimpse_key: Optional[torch.Tensor] = None
    glimpse_val: Optional[torch.Tensor] = None
    logit_key: Optional[torch.Tensor] = None

    def __getitem__(self, key: Union[int, slice, torch.Tensor]) -> AttentionDecoderCache:
        """Slice the cache to select specific batch elements.

        This method allows efficient batching and sub-batching during decoding.
        All tensor fields are sliced along the batch dimension (dimension 0).

        Args:
            key (Union[int, slice, torch.Tensor]): Index, slice, or mask for selection.

        Returns:
            AttentionDecoderCache: New cache instance with sliced tensors.
        """
        # Handle integer indexing (add batch dimension back)
        if isinstance(key, int):
            return AttentionDecoderCache(
                node_embeddings=self.node_embeddings[key].unsqueeze(0),
                graph_context=(self.graph_context[key].unsqueeze(0) if self.graph_context is not None else None),
                glimpse_key=(self.glimpse_key[key].unsqueeze(0) if self.glimpse_key is not None else None),
                glimpse_val=(self.glimpse_val[key].unsqueeze(0) if self.glimpse_val is not None else None),
                logit_key=(self.logit_key[key].unsqueeze(0) if self.logit_key is not None else None),
            )

        # Handle slice or tensor indexing (preserves batch dimension)
        return AttentionDecoderCache(
            node_embeddings=self.node_embeddings[key],
            graph_context=self.graph_context[key] if self.graph_context is not None else None,
            glimpse_key=self.glimpse_key[key] if self.glimpse_key is not None else None,
            glimpse_val=self.glimpse_val[key] if self.glimpse_val is not None else None,
            logit_key=self.logit_key[key] if self.logit_key is not None else None,
        )

    @property
    def context_node_projected(self) -> Optional[torch.Tensor]:
        """Alias for graph_context (backwards compatibility).

        Returns:
            Optional[torch.Tensor]: Same as self.graph_context.
        """
        return self.graph_context

    @context_node_projected.setter
    def context_node_projected(self, value: Optional[torch.Tensor]) -> None:
        """Setter for context_node_projected (backwards compatibility).

        Args:
            value (Optional[torch.Tensor]): New graph context tensor.
        """
        self.graph_context = value
