"""Generic context embedding module.

This module provides the GenericContextEmbedder, which serves as a fallback
for routing problems that do not require specialized state feature extraction.

Attributes:
    GenericContextEmbedder: Basic context embedder using linear projections.

Example:
    >>> from logic.src.models.subnets.embeddings.context.generic import GenericContextEmbedder
    >>> embedder = GenericContextEmbedder(embed_dim=128)
    >>> initial_h = embedder(nodes)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from logic.src.constants.models import NODE_DIM

from .base import ContextEmbedder


class GenericContextEmbedder(ContextEmbedder):
    """Generic context embedder for baseline routing problems.

    Provides simple linear projections for depot and node locations, and
    uses the current node embedding as the dynamic step context.

    Attributes:
        init_embed (nn.Linear): Projection for non-depot node features.
        init_embed_depot (nn.Linear): Specialized projection for depot location.
        project_step_context (nn.Module): Projection layer for dynamic step context.
    """

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0) -> None:
        """Initializes GenericContextEmbedder.

        Args:
            embed_dim: Internal embedding dimensionality.
            node_dim: Input node feature dimension.
            temporal_horizon: Ignored in generic implementation.
        """
        super().__init__(embed_dim, node_dim, temporal_horizon)
        self.init_embed = nn.Linear(node_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

        # Step context projection: usually just current node for generic
        # If step_context_dim == embed_dim, it's just the gathered embedding
        self.project_step_context = (
            nn.Identity() if self.step_context_dim == embed_dim else nn.Linear(self.step_context_dim, embed_dim)
        )

    def init_node_embeddings(self, nodes: Dict[str, Any]) -> torch.Tensor:
        """Projects raw problem instance locations into embedding space.

        Args:
            nodes: Instance dictionary containing 'depot' and 'loc'/'locs'.

        Returns:
            torch.Tensor: Combined node embeddings (depot + others).
        """
        # Fallback to 'loc' or 'locs'
        locs_key = "locs" if "locs" in nodes else "loc"
        node_features = nodes[locs_key]

        return torch.cat(
            (
                self.init_embed_depot(nodes["depot"])[:, None, :],
                self.init_embed(node_features),  # type: ignore[misc]
            ),
            1,
        )

    def _step_context(self, embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """Extracts current node embedding as the step context.

        Args:
            embeddings: Current node embeddings.
            state: Environment state object.

        Returns:
            torch.Tensor: Projected current node embedding.
        """
        current_node = state.get_current_node() if hasattr(state, "get_current_node") else state.get("current_node")
        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)

        batch_size = current_node.size(0)

        if embeddings.size(0) != batch_size:
            if embeddings.size(0) == 1:
                embeddings = embeddings.expand(batch_size, -1, -1)
            elif batch_size % embeddings.size(0) == 0:
                embeddings = embeddings.repeat_interleave(batch_size // embeddings.size(0), dim=0)
            else:
                embeddings = embeddings.expand(batch_size, -1, -1)

        # Gather current node embedding: [batch, 1, embed_dim]
        step_context = embeddings.gather(
            1,
            current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim),
        )

        return self.project_step_context(step_context)

    @property
    def step_context_dim(self) -> int:
        """Returns the dimensionality of the generic current-node context.

        Returns:
            int: Size of the embedding dimension.
        """
        return self.embed_dim
