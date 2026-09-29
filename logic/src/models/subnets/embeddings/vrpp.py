"""VRPP Embedding module.

This module provides the VRPPInitEmbedding layer, which encodes locations
and waste levels for Vehicle Routing Problems with Profits.

Attributes:
    VRPPInitEmbedding: Initial feature encoder for VRPP instances.

Example:
    >>> from logic.src.models.subnets.embeddings.vrpp import VRPPInitEmbedding
    >>> embed = VRPPInitEmbedding(embed_dim=128)
    >>> h = embed(td)
"""

from __future__ import annotations

from typing import Any, Dict, Union

import torch
from tensordict import TensorDict
from torch import nn


class VRPPInitEmbedding(nn.Module):
    """Initial embedding for VRPP problems.

    Projects static node features (coordinates and container waste quantities)
    into a high-dimensional embedding space.

    Attributes:
        node_embed (nn.Linear): Linear projection for customer node features.
        depot_embed (nn.Linear): Specialized linear projection for depot location.
    """

    def __init__(
        self, embed_dim: int = 128, node_dim: int = 3, temporal_horizon: int = 0, legacy_depot_projection: bool = False
    ) -> None:
        """Initializes VRPPInitEmbedding.

        Args:
            embed_dim: Internal embedding dimensionality.
            node_dim: Base dimensionality of customer node features.
            temporal_horizon: Window size for temporal features (if > 0).
            legacy_depot_projection: Use the historical AttentionModel node projection for a concatenated depot.
        """
        super().__init__()
        self.legacy_depot_projection = legacy_depot_projection
        self.node_dim = node_dim
        self.temporal_horizon = temporal_horizon
        input_dim = node_dim + (temporal_horizon if temporal_horizon > 0 else 0)
        # Node features: x, y, waste (+ temporal_features if temporal_horizon > 0)
        self.node_embed = nn.Linear(input_dim, embed_dim)
        # Depot features: x, y
        self.depot_embed = nn.Linear(2, embed_dim)

    def forward(self, td: Union[TensorDict, Dict[str, Any]], temporal_features: bool = True) -> torch.Tensor:
        """Encodes VRPP instance features into initial node embeddings.

        Args:
            td: TensorDict or dict containing instance metadata ('locs', 'depot', 'waste').
            temporal_features: Whether to include temporal features if horizon > 0.

        Returns:
            torch.Tensor: Initial node embeddings of shape (batch, num_nodes, dim).
        """
        locs = td.get("locs") if hasattr(td, "get") and "locs" in td.keys() else None
        if locs is None:
            locs = td.get("loc") if hasattr(td, "get") else None
        if locs is None:
            locs = td["locs"]
        depot = td["depot"]
        waste = td.get("waste") if hasattr(td, "get") else None
        batch_size = td.batch_size[0] if isinstance(td, TensorDict) else locs.size(0)
        device = td.device if isinstance(td, TensorDict) else locs.device

        if waste is None:
            waste = torch.zeros(batch_size, locs.size(1), device=device)
        elif waste.dim() == 1:
            waste = waste.unsqueeze(0)

        node_features_list: list[torch.Tensor] = [locs, waste.unsqueeze(-1)]
        if self.temporal_horizon > 0 and temporal_features:
            temp_feat = td.get("temporal_features") if hasattr(td, "get") else None
            if temp_feat is not None:
                node_features_list.append(temp_feat)
            else:
                node_features_list.append(
                    torch.zeros(
                        locs.size(0),
                        locs.size(1),
                        self.temporal_horizon,
                        device=locs.device,
                    )
                )

        node_features = torch.cat(node_features_list, dim=-1)

        # Check if locs already contains depot (e.g. RL4CO environments)
        # or if depot is separate (legacy problem instances)
        depot_in_locs = locs.size(1) == waste.size(1) + 1 or (
            locs.size(1) == waste.size(1) and torch.allclose(locs[:, 0, :], depot, atol=1e-4)
        )

        if not depot_in_locs:
            # Traditional: separate depot and customers
            customer_emb = self.node_embed(node_features)
            depot_emb = self.depot_embed(depot).unsqueeze(1)
            return torch.cat([depot_emb, customer_emb], dim=1)

        embeddings = self.node_embed(node_features)
        if not self.legacy_depot_projection:
            embeddings[:, 0, :] = self.depot_embed(depot)
        return embeddings

    def init_node_embeddings(self, nodes: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Alias for forward to support legacy context embedder interface."""
        return self.forward(nodes, *args, **kwargs)
