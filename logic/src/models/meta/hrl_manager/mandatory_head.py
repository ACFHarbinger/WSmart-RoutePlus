"""Packing Head for Node Categorization.

This module provides the `PackingHead`, which determines if a node must
be collected in the current period based on local and spatial context.

Attributes:
    PackingHead: MLP-based binary categorical selection head.

Example:
    None
"""

from __future__ import annotations

import torch
from torch import nn


class PackingHead(nn.Module):
    """MLP head for per-node mandatory status selection.

    Produces categorical logits for each node independently, indicating whether
    the node is 'optional' (opportunistic collection) or 'mandatory' (must collect).

    Attributes:
        net (nn.Sequential): MLP layers (Linear -> ReLU -> Linear).
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initializes the selection head.

        Args:
            input_dim: size of node embeddings (spatial + temporal fused).
            hidden_dim: width of the intermediate representation.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predicts selection logits per node.

        Args:
            x: local node embeddings [B, N, D].

        Returns:
            torch.Tensor: logits for [Optional, Mandatory] [B, N, 2].
        """
        return self.net(x)
