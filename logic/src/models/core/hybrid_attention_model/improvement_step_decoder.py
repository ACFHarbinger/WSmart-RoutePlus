"""Improvement Step Decoder for Hybrid Models.

This module provides the `ImprovementStepDecoder`, which predicts the next
neighborhood search operator to apply based on the current graph and solution
context.

Attributes:
    ImprovementStepDecoder: Neural selector for local search moves.

Example:
    >>> decoder = ImprovementStepDecoder(embed_dim=128, n_operators=5)
    >>> logits, mask = decoder(td, embeddings, env)
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.envs.base.base import RL4COEnvBase
from logic.src.models.common.improvement.decoder import ImprovementDecoder


class ImprovementStepDecoder(ImprovementDecoder):
    """Decoder mapping graph context to operator scores.

    Processes global node embeddings into a preference distribution over a fixed
    set of vectorized local search heuristics.

    Attributes:
        output_head (nn.Sequential): MLP mapping features to operator logits.
        context_projection (nn.Linear): Transformation for pooled graph features.
    """

    def __init__(self, embed_dim: int = 128, n_operators: int = 6, hidden_dim: int = 128) -> None:
        """Initializes the improvement step decoder.

        Args:
            embed_dim: Width of input graph embeddings.
            n_operators: Count of registered local search moves.
            hidden_dim: MLP bottleneck dimension.
        """
        super().__init__(embed_dim=embed_dim)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_operators),
        )
        self.context_projection = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        td: TensorDict,
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        env: Optional[RL4COEnvBase] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates operator selection logits from graph context.

        Args:
            td: TensorDict containing the current problem and solution state.
            embeddings: Node-level embeddings from the encoder [B, N, D].
            env: The environment defining the search operators.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logits: Unnormalized operator scores [B, n_operators].
                - mask: Boolean availability mask (currently all True).
        """
        if isinstance(embeddings, tuple):
            embeddings = embeddings[0]

        # Extract global graph context via mean pooling
        graph_context = embeddings.mean(dim=1)

        # Map to operator distribution
        context = self.context_projection(graph_context)
        logits = self.output_head(context)

        # Default to all operators being feasible
        mask = torch.zeros_like(logits, dtype=torch.bool)

        return logits, mask
