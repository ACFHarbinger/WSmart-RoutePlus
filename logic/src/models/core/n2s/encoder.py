"""N2S Encoder implementation.

This module provides the `N2SEncoder`, which implements a lightweight
Transformer with neighborhood masking. It attends only to the K-nearest
neighbors of each node to maintain efficiency and focus on local moves.

Attributes:
    N2SEncoder: Neural encoder for neighborhood-constrained attention.

Example:
    >>> encoder = N2SEncoder(embed_dim=128, k_neighbors=20)
    >>> h = encoder(td)
"""

from __future__ import annotations

from typing import Any, cast

import torch
from tensordict import TensorDict
from torch import nn

from logic.src.models.common.improvement.encoder import ImprovementEncoder
from logic.src.models.subnets.modules import MultiHeadAttention, Normalization


class N2SEncoder(ImprovementEncoder):
    """N2S Encoder with sparse neighborhood attention.

    Optimizes the representation of problem nodes by restricting the attention
    mechanism to a local k-neighbor window, facilitating more efficient
    iterative enhancement.

    Attributes:
        k_neighbors (int): number of spatially proximal nodes to attend to.
        num_heads (int): count of attention heads.
        mha (MultiHeadAttention): masked attention subnet.
        norm (Normalization): layer normalization module.
        ff (nn.Sequential): feed-forward expansion network.
        init_proj (nn.Linear): Initial spatial coordinate projector.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        k_neighbors: int = 20,
        **kwargs: Any,
    ) -> None:
        """Initializes the N2S encoder.

        Args:
            embed_dim: Dimensionality of the node embeddings.
            num_heads: Number of attention heads.
            k_neighbors: Number of nearest neighbors to attend to.
            kwargs: Additional keyword arguments.
        """
        super().__init__(embed_dim=embed_dim)
        self.k_neighbors = k_neighbors
        self.num_heads = num_heads

        self.mha = MultiHeadAttention(num_heads, embed_dim, embed_dim)
        self.norm = Normalization(embed_dim, "layer")
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Explicitly initialize projection to avoid lazy init in forward
        self.init_proj = nn.Linear(2, embed_dim)

    def _get_neighborhood_mask(self, td: TensorDict) -> torch.Tensor:
        """Computes a binary mask based on spatial proximity.

        Args:
            td: Environment state containing node coordinates ('locs' and 'depot').

        Returns:
            torch.Tensor: Boolean mask where True indicates an excluded neighbor [B, N, N].
        """
        # locs: [bs, n, 2]
        depot = cast(torch.Tensor, td["depot"]).unsqueeze(1)  # TensorDict values are a tensor union
        customers = cast(torch.Tensor, td["locs"])
        locs = torch.cat([depot, customers], dim=1)

        # dist: [bs, n, n]
        dist = torch.cdist(locs, locs)

        # Get indices of K-nearest neighbors
        # (self is always included in nearest)
        _, topk_indices = dist.topk(self.k_neighbors + 1, largest=False, sorted=False)

        # Create mask: True for excluded neighbors
        bs, n, _ = dist.shape
        mask = torch.ones(bs, n, n, device=dist.device, dtype=torch.bool)
        mask.scatter_(2, topk_indices, False)

        return mask

    def forward(self, td: TensorDict, **kwargs: Any) -> torch.Tensor:
        """Encodes node features using neighborhood-constrained attention.

        Args:
            td: TensorDict containing "depot" and "locs" keys.
            kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Refined node embeddings [B, N, embed_dim].
        """
        # Initial embedding (coords)
        depot = cast(torch.Tensor, td["depot"]).unsqueeze(1)  # TensorDict values are a tensor union
        customers = cast(torch.Tensor, td["locs"])
        h_raw = torch.cat([depot, customers], dim=1)  # [bs, n, 2]

        h = self.init_proj(h_raw)

        # Neighborhood mask
        mask = self._get_neighborhood_mask(td)

        # Transformer layer
        attn_out = self.mha(h, mask=mask)
        h = self.norm(h + attn_out)
        h = h + self.ff(h)

        return h
