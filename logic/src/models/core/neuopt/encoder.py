"""NeuOpt Encoder implementation.

This module provides the `NeuOptEncoder`, a deep Transformer-based architecture
designed to encode the spatial and topological state of a combinatorial problem
for iterative optimization.

Attributes:
    NeuOptEncoder: Transformer encoder for global problem state representation.

Example:
    >>> encoder = NeuOptEncoder(embed_dim=128)
    >>> h = encoder(td)
"""

from __future__ import annotations

from typing import Any, cast

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.improvement.encoder import ImprovementEncoder
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization


class NeuOptEncoder(ImprovementEncoder):
    """NeuOpt Encoder for global solution state representation.

    Iteratively refines node embeddings using a stack of self-attention layers
    to capture long-range dependencies and spatial relationships within the
    routing problem.

    Attributes:
        num_layers (int): depth of the Transformer stack.
        init_proj (nn.Linear): Linear projection for raw coordinates.
        layers (nn.ModuleList): list of self-attention and feed-forward blocks.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initializes the NeuOpt encoder.

        Args:
            embed_dim: Dimensionality of the node embeddings.
            num_heads: Number of attention heads.
            num_layers: Number of Transformer blocks.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.num_layers = num_layers

        # Initial projection for coordinates
        self.init_proj = nn.Linear(2, embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "mha": MultiHeadAttention(num_heads, embed_dim, embed_dim),
                        "norm1": Normalization(embed_dim, "layer"),
                        "ff": nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 4),
                            nn.ReLU(),
                            nn.Linear(embed_dim * 4, embed_dim),
                        ),
                        "norm2": Normalization(embed_dim, "layer"),
                    }
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, td: TensorDict, **kwargs: Any) -> torch.Tensor:
        """Encodes the problem instance into high-dimensional node features.

        Args:
            td: TensorDict containing "depot" and "locs" keys.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Encoded node embeddings [B, N, embed_dim].
        """
        depot = cast(torch.Tensor, td["depot"]).unsqueeze(1)  # TensorDict values are a tensor union
        customers = cast(torch.Tensor, td["locs"])  # TensorDict values are a tensor union
        h = torch.cat([depot, customers], dim=1)  # [bs, n, 2]

        h = self.init_proj(h)

        for layer_module in self.layers:
            # Cast to ModuleDict for indexing
            layer = cast(nn.ModuleDict, layer_module)
            # Multi-head attention
            attn_out = layer["mha"](h)
            h = layer["norm1"](h + attn_out)

            # Feed-forward
            ff_out = layer["ff"](h)
            h = layer["norm2"](h + ff_out)

        return h
