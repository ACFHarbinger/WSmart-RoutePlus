"""Abstract base class for context embeddings.

This module provides the ContextEmbedder abstract base class, which defines
the interface for problem-specific step context extraction during decoding.

Attributes:
    ContextEmbedder: Abstract base class for context embedding modules.

Example:
    >>> class MyContextEmbedder(ContextEmbedder):
    ...     @property
    ...     def step_context_dim(self): return 128
    ...     def init_node_embeddings(self, nodes): return ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import torch
from torch import nn


class ContextEmbedder(nn.Module, ABC):
    """Abstract base class for problem-specific context embeddings.

    Responsible for initializing node embeddings from raw features and
    extracting dynamic step context during autoregressive decoding.

    Attributes:
        embed_dim (int): Dimensionality of node embeddings.
        node_dim (int): Dimensionality of raw input node features.
        temporal_horizon (int): Maximum sequence length or time horizon.
        init_embed (Optional[nn.Module]): Registered initial embedding module.
    """

    def __init__(self, embed_dim: int, node_dim: int, temporal_horizon: int) -> None:
        """Initializes the ContextEmbedder.

        Args:
            embed_dim: Internal embedding dimensionality.
            node_dim: Input feature dimensionality.
            temporal_horizon: Max steps or time limit.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.node_dim = node_dim
        self.temporal_horizon = temporal_horizon
        self.init_embed: Optional[nn.Module] = None

    @abstractmethod
    def init_node_embeddings(self, nodes: Dict[str, Any]) -> torch.Tensor:
        """Initializes node embeddings from raw input features.

        Args:
            nodes: Dictionary containing raw problem instance features.

        Returns:
            torch.Tensor: Initial node embeddings.
        """
        pass

    @property
    @abstractmethod
    def step_context_dim(self) -> int:
        """Gets the dimensionality of the step context embedding.

        Returns:
            int: Feature dimension size.
        """
        pass

    def forward(
        self,
        nodes_or_embeddings: Union[torch.Tensor, Dict[str, Any]],
        state: Optional[Any] = None,
    ) -> torch.Tensor:
        """Forward pass handling both initialization and step context extraction.

        Args:
            nodes_or_embeddings: Raw node features (if state is None) or existing
                node embeddings (during decoding steps).
            state: Optional state object for dynamic context derivation.

        Returns:
            torch.Tensor: Initial node embeddings or current step context.
        """
        if state is None:
            return self.init_node_embeddings(nodes_or_embeddings)  # type: ignore[arg-type]
        return self._step_context(nodes_or_embeddings, state)  # type: ignore[arg-type]

    def _step_context(self, embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """Extracts the default step context based on current node position.

        Args:
            embeddings: Current node embeddings.
            state: Current environment state object.

        Returns:
            torch.Tensor: Current node embedding of shape (batch, 1, dim).
        """
        current_node = state.get_current_node() if hasattr(state, "get_current_node") else state.get("current_node")

        # Ensure current_node is (batch)
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

        return step_context
