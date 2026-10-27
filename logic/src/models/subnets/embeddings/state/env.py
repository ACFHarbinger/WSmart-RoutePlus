"""Environment state embedding base module.

This module provides the EnvState base class, which handles the extraction and
projection of dynamic context from environment states during the decoding phase.

Attributes:
    EnvState: Abstract base class for problem-specific state representations.

Example:
    >>> from logic.src.models.subnets.embeddings.state.env import EnvState
    >>> state_embedder = EnvState(embed_dim=128)
    >>> context = state_embedder(embeddings, td)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn


class EnvState(nn.Module):
    """Base class for environment context embeddings.

    Provides a standard interface for extracting the current node's embedding
    and combining it with auxiliary state features (e.g., remaining capacity).

    Attributes:
        embed_dim (int): Dimensionality of the joint embedding space.
        project_context (Optional[nn.Linear]): Layer to project raw state features.
    """

    project_context: Optional[nn.Linear]

    def __init__(self, embed_dim: int, step_context_dim: int = 0, node_dim: int = 0) -> None:
        """Initializes EnvState.

        Args:
            embed_dim: Target embedding dimensionality.
            step_context_dim: Raw dimension of state-specific metadata.
            node_dim: Base dimensionality of individual node features (unused).
        """
        super().__init__()
        self.embed_dim = embed_dim
        if step_context_dim > 0:
            self.project_context = nn.Linear(step_context_dim, embed_dim)
        else:
            self.project_context = None

    def forward(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        """Combines current node embedding with projected state features.

        Args:
            embeddings: Current latent node features (batch, nodes, dim).
            td: Current problem state / metadata container.

        Returns:
            torch.Tensor: Combined context vector of shape (batch, dim).
        """
        cur_node_embedding = self._cur_node_embedding(embeddings, td)
        state_embedding = self._state_embedding(embeddings, td)

        if self.project_context is not None:
            return cur_node_embedding + self.project_context(state_embedding)

        return cur_node_embedding + state_embedding

    def _cur_node_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        """Extracts the embedding of the currently active node.

        Args:
            embeddings: Lattice of node embeddings (batch, nodes, dim).
            td: Environment state containing "current_node" index.

        Returns:
            torch.Tensor: Selected node embeddings (batch, dim).
        """
        # Current node is usually the last visited node
        cur_node = td["current_node"]
        if cur_node.dim() == 1:
            cur_node = cur_node[:, None]
        return torch.gather(embeddings, 1, cur_node.expand(-1, embeddings.size(-1))[:, None, :]).squeeze(1)

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        """Provides problem-specific state level context features.

        Args:
            embeddings: Existing node features (used for metadata extraction).
            td: Environment state container.

        Returns:
            torch.Tensor: Static zero-vector by default. Shape (batch, dim).
        """
        return torch.zeros(embeddings.size(0), self.embed_dim, device=embeddings.device)
