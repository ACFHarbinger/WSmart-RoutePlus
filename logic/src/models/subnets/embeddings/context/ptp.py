"""PTP specific context embedding module.

This module provides the PTPContextEmbedder, which handles initial state
projections and dynamic context extraction for Vehicle Routing Problems with Profits.

Attributes:
    PTPContextEmbedder: Context embedder tailored for PTP/CVRP variants.

Example:
    >>> from logic.src.models.subnets.embeddings.context.ptp import PTPContextEmbedder
    >>> embedder = PTPContextEmbedder(embed_dim=128)
    >>> context = embedder(embeddings, state)
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from logic.src.constants.models import NODE_DIM, PTP_STEP_CONTEXT_OFFSET

from .base import ContextEmbedder


class PTPContextEmbedder(ContextEmbedder):
    """Context embedder for Profitable Tour Problem (PTP) and CVRP families.

    Handles initial node features (locations + waste/demand) and extracts
    dynamic step features such as current vehicle load or remaining capacity.

    Attributes:
        init_embed (nn.Linear): Projection for non-depot customer features.
        init_embed_depot (nn.Linear): Projection for depot location.
        project_step_context (nn.Linear): Projection for fused step context.
    """

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0) -> None:
        """Initializes PTPContextEmbedder.

        Args:
            embed_dim: Target embedding dimensionality.
            node_dim: Base dimensionality of node features.
            temporal_horizon: Size of temporal feature window (if any).
        """
        super().__init__(embed_dim, node_dim, temporal_horizon)
        input_dim = node_dim
        if temporal_horizon > 0:
            input_dim += temporal_horizon

        self.init_embed = nn.Linear(input_dim, embed_dim)
        self.init_embed_depot = nn.Linear(2, embed_dim)

        # Step context projection: current_node_embed + state_features
        self.project_step_context = nn.Linear(self.step_context_dim, embed_dim)

    def init_node_embeddings(self, nodes: Dict[str, Any], temporal_features: bool = True) -> torch.Tensor:
        """Initializes node embeddings from locations and problem-specific features.

        Args:
            nodes: Input dictionary containing 'locs'/'loc', 'waste'/'demand', 'depot'.
            temporal_features: Whether to include temporal features if horizon > 0.

        Returns:
            torch.Tensor: Initial node embeddings including the depot.
        """
        # Initial node embedding: locs + waste
        # Waste is usually [batch_size, num_nodes]
        # locs is [batch_size, num_nodes, 2]

        locs = nodes.get("locs") if "locs" in nodes.keys() else nodes.get("loc")
        if locs is None:
            # Fallback for missing locs
            return torch.zeros(
                1,
                1,
                self.embed_dim,
                device=nodes.device if hasattr(nodes, "device") else None,
            )

        # Use waste for node quantities
        waste = nodes.get("waste")
        if waste is None:
            waste = torch.zeros(locs.shape[0], locs.shape[1], 1, device=locs.device) if self.node_dim > 2 else None

        node_features: list[torch.Tensor] = [locs]
        if waste is not None:
            if waste.dim() == 2:
                waste = waste.unsqueeze(-1)
            node_features.append(waste)

        # Add temporal features if available
        if self.temporal_horizon > 0 and temporal_features:
            temp_feat = nodes.get("temporal_features")
            if temp_feat is not None:
                node_features.append(temp_feat)
            else:
                # Pad with zeros if missing
                node_features.append(
                    torch.zeros(
                        locs.size(0),
                        locs.size(1),
                        self.temporal_horizon,
                        device=locs.device,
                    )
                )

        node_features_tensor = torch.cat(node_features, -1)

        # Determine if locs already contains the depot
        depot = nodes["depot"]
        waste_raw = nodes.get("waste")

        # Check if already concatenated: shape matches OR first element matches depot
        is_concatenated = waste_raw is not None and (
            locs.size(1) == waste_raw.size(1) + 1
            or (locs.size(1) == waste_raw.size(1) and torch.allclose(locs[..., 0, :], depot, atol=1e-4))
        )

        if is_concatenated:
            return self.init_embed(node_features_tensor)  # type: ignore[misc]

        # Traditional: separate depot and customers
        return torch.cat(
            (
                self.init_embed_depot(depot)[:, None, :],
                self.init_embed(node_features_tensor),  # type: ignore[misc]
            ),
            dim=1,
        )

    def _step_context(self, embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """Extracts current PTP step context.

        Fuses three signals into the decoder query at each selection step:
        1. Current node embedding (where the vehicle is).
        2. Sum of waste on unvisited nodes (remaining profit on the map).
        3. Mean Euclidean distance from the current node to all unvisited
           profitable nodes (spatial spread of remaining profit).

        These two scalars together give the decoder a rich signal to decide
        whether continued travel is profitable or early termination is optimal.
        The PTP has no capacity constraint, so load state is omitted here.

        Tensor shapes:
            current_node_embed  : [B, 1, embed_dim]
            unvisited_waste_sum : [B, 1, 1]
            mean_dist_unvisited : [B, 1, 1]
            combined            : [B, 1, embed_dim + 2]  → [B, 1, embed_dim]

        TensorDict keys consumed:
            ``waste``        : [B, N]  — waste per node (depot waste = 0)
            ``visited``      : [B, N]  — visit flags
            ``locs``         : [B, N, 2] — Euclidean coordinates (fallback)
            ``current_node`` : [B] or [B, 1] — current vehicle position index

        Args:
            embeddings: Current node embeddings, shape (batch, num_nodes, embed_dim).
            state: Current environment state (TensorDict or compatible).

        Returns:
            torch.Tensor: Projected step context embedding, shape (batch, 1, embed_dim).
        """
        current_node = state.get_current_node() if hasattr(state, "get_current_node") else state.get("current_node")
        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)  # [B]

        batch_size = current_node.size(0)
        if embeddings.size(0) == 1 and batch_size > 1:
            embeddings = embeddings.expand(batch_size, -1, -1)

        # 1. Current node embedding [B, 1, embed_dim]
        current_node_embed = embeddings.gather(
            1,
            current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim),
        )

        # Gather shared state tensors
        td = getattr(state, "td", state)
        waste = td.get("waste", None)  # [B, N] or [B, N-1]
        visited = td.get("visited", None)  # [B, N]
        locs = td.get("locs", None)  # [B, N, 2]

        # Align waste to visited shape (depot column may be absent)
        if waste is not None and visited is not None and waste.shape[-1] < visited.shape[-1]:
            waste = torch.cat([torch.zeros(batch_size, 1, device=embeddings.device), waste], dim=-1)

        # 2. Mean of waste on unvisited nodes [B, 1, 1]
        if waste is not None and visited is not None:
            unvisited_mask = (~visited).float()  # [B, N]
            num_nodes = waste.shape[-1]
            unvisited_waste_mean = (waste * unvisited_mask).sum(dim=-1, keepdim=True).unsqueeze(1) / num_nodes
        else:
            unvisited_waste_mean = torch.zeros(batch_size, 1, 1, device=embeddings.device)

        # 3. Mean Euclidean distance from current node to unvisited profitable nodes [B, 1, 1]
        if locs is not None and visited is not None and waste is not None:
            # Current position: [B, 2]
            cur_idx = current_node.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, 2)
            cur_pos = locs.gather(1, cur_idx).squeeze(1)  # [B, 2]

            # Distance from current to all nodes: [B, N]
            dist_to_all = torch.norm(locs - cur_pos.unsqueeze(1), dim=-1)

            # Mask: unvisited AND have non-zero waste (exclude depot at idx 0)
            unvisited_profitable = (~visited) & (waste > 0)  # [B, N]
            unvisited_profitable[:, 0] = False  # exclude depot

            count = unvisited_profitable.float().sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
            mean_dist = (dist_to_all * unvisited_profitable.float()).sum(dim=-1, keepdim=True) / count
            mean_dist_unvisited = mean_dist.unsqueeze(1)  # [B, 1, 1]
        else:
            mean_dist_unvisited = torch.zeros(batch_size, 1, 1, device=embeddings.device)

        # Concat all signals and project [B, 1, embed_dim+2] → [B, 1, embed_dim]
        combined = torch.cat([current_node_embed, unvisited_waste_mean, mean_dist_unvisited], dim=-1)
        return self.project_step_context(combined)

    @property
    def step_context_dim(self) -> int:
        """Gets the dimensionality of the fused PTP step context.

        Aggregates node embedding dim and step-specific metadata offset.

        Returns:
            int: Dimension size.
        """
        return self.embed_dim + PTP_STEP_CONTEXT_OFFSET
