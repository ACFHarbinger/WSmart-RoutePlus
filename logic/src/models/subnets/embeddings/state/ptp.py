"""PTP specific context embedding module.

This module provides the PTPState component, which extracts problem-specific
metadata like remaining tour length or capacity for the PTP context.

Attributes:
    PTPState: State encoder for Vehicle Routing Problems with Profits.

Example:
    >>> from logic.src.models.subnets.embeddings.state.ptp import PTPState
    >>> state_embedder = PTPState(embed_dim=128)
    >>> context = state_embedder(embeddings, td)
"""

from __future__ import annotations

from typing import Any

import torch

from .env import EnvState


class PTPState(EnvState):
    """Context embedding for PTP.

    Evolves the environment context by incorporating dynamic features such
    as the remaining allowed tour length or available vehicle capacity.

    Attributes:
        embed_dim (int): Dimensionality of the projected state context.
    """

    def __init__(self, embed_dim: int) -> None:
        """Initializes PTPState.

        Args:
            embed_dim: Resulting embedding dimensionality.
        """
        super().__init__(embed_dim, step_context_dim=1)

    def _state_embedding(self, embeddings: torch.Tensor, td: Any) -> torch.Tensor:
        """Extracts remaining resources (length/capacity) from the state.

        Args:
            embeddings: Current node embeddings (used for device/batch info).
            td: Environment state dictionary/container.

        Returns:
            torch.Tensor: Normalized state features of shape (batch, 1).
        """
        # Remaining length or capacity
        # We use 'remaining_length' or 'remaining_capacity'
        if "remaining_length" in td.keys():
            state = td["remaining_length"]
        elif "remaining_capacity" in td.keys():
            state = td["remaining_capacity"]
        else:
            state = torch.ones(embeddings.size(0), device=embeddings.device)

        if state.dim() == 1:
            state = state[:, None]

        return state
