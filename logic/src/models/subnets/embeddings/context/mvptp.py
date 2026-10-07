"""MVPTP-specific context embedding module.

This module provides ``MVPTPContextEmbedder``, a capacity-aware extension of
``PTPContextEmbedder`` for the Multi-Vehicle Profitable Tour Problem (MVPTP).

While the uncapacitated PTP context uses (unvisited waste, mean travel distance)
to guide early termination, the MVPTP additionally exposes the vehicle's remaining
capacity so the decoder can learn to stop visiting bins when the vehicle is nearly
full even if profitable nodes remain.

Attributes:
    MVPTPContextEmbedder: Context embedder for MVPTP / orienteering with capacity.

Example:
    >>> from logic.src.models.subnets.embeddings.context.mvptp import MVPTPContextEmbedder
    >>> embedder = MVPTPContextEmbedder(embed_dim=128)
    >>> context = embedder(embeddings, state)   # [B, 1, 128]
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from logic.src.constants.models import MVPTP_STEP_CONTEXT_OFFSET, NODE_DIM

from .base import ContextEmbedder


class MVPTPContextEmbedder(ContextEmbedder):
    """Context embedder for Multi-Vehicle Profitable Tour Problem (MVPTP).

    Extends the pure-PTP context with a vehicle remaining-capacity scalar,
    giving the decoder three dynamic signals per decoding step:

    1. **Unvisited waste sum** — total profit still available on the map.
    2. **Mean distance to unvisited profitable nodes** — spatial cost signal.
    3. **Remaining capacity** — hard constraint signal.

    This combination lets the policy learn: "is there still enough space and
    enough nearby profit to justify continuing the route?"

    Attributes:
        init_embed (nn.Linear): Projection for non-depot customer features.
        init_embed_depot (nn.Linear): Projection for depot location.
        project_step_context (nn.Linear): Projection for fused step context.
    """

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0) -> None:
        """Initializes MVPTPContextEmbedder.

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

        # Step context projection: embed_dim + MVPTP_STEP_CONTEXT_OFFSET → embed_dim
        self.project_step_context = nn.Linear(self.step_context_dim, embed_dim)

    def init_node_embeddings(self, nodes: Dict[str, Any], temporal_features: bool = True) -> torch.Tensor:
        """Initializes node embeddings from locations and problem-specific features.

        Delegates to the same logic used by ``PTPContextEmbedder``: projects
        ``[locs, waste]`` per customer node and the depot separately.

        Args:
            nodes: Input dictionary containing ``locs``, ``waste``, ``depot``.
            temporal_features: Whether to include temporal features if horizon > 0.

        Returns:
            torch.Tensor: Initial node embeddings including the depot.
        """
        locs = nodes.get("locs") if "locs" in nodes.keys() else nodes.get("loc")
        if locs is None:
            return torch.zeros(
                1,
                1,
                self.embed_dim,
                device=nodes.device if hasattr(nodes, "device") else None,
            )

        waste = nodes.get("waste")
        if waste is None:
            waste = torch.zeros(locs.shape[0], locs.shape[1], 1, device=locs.device) if self.node_dim > 2 else None

        node_features: list[torch.Tensor] = [locs]
        if waste is not None:
            if waste.dim() == 2:
                waste = waste.unsqueeze(-1)
            node_features.append(waste)

        if self.temporal_horizon > 0 and temporal_features:
            temp_feat = nodes.get("temporal_features")
            if temp_feat is not None:
                node_features.append(temp_feat)
            else:
                node_features.append(torch.zeros(locs.size(0), locs.size(1), self.temporal_horizon, device=locs.device))

        node_features_tensor = torch.cat(node_features, -1)

        depot = nodes["depot"]
        waste_raw = nodes.get("waste")
        is_concatenated = waste_raw is not None and (
            locs.size(1) == waste_raw.size(1) + 1
            or (locs.size(1) == waste_raw.size(1) and torch.allclose(locs[..., 0, :], depot, atol=1e-4))
        )

        if is_concatenated:
            return self.init_embed(node_features_tensor)  # type: ignore[misc]

        return torch.cat(
            (
                self.init_embed_depot(depot)[:, None, :],
                self.init_embed(node_features_tensor),  # type: ignore[misc]
            ),
            dim=1,
        )

    def _step_context(self, embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """Extracts current MVPTP step context.

        Fuses four signals at each decoding step:
        1. Current node embedding.
        2. Sum of waste on unvisited nodes (remaining profit signal).
        3. Mean Euclidean distance to unvisited profitable nodes (travel cost signal).
        4. Remaining vehicle capacity (hard constraint signal).

        Tensor shapes:
            current_node_embed  : [B, 1, embed_dim]
            unvisited_waste_sum : [B, 1, 1]
            mean_dist_unvisited : [B, 1, 1]
            remaining_capacity  : [B, 1, 1]
            combined            : [B, 1, embed_dim + 3]  → [B, 1, embed_dim]

        TensorDict keys consumed:
            ``waste``               : [B, N]    — waste per node (depot = 0)
            ``visited``             : [B, N]    — visit flags
            ``locs``                : [B, N, 2] — Euclidean coordinates
            ``remaining_capacity``  : [B]       — residual vehicle capacity
            ``current_load``        : [B]       — alternative capacity key
            ``current_node``        : [B] or [B, 1]

        Args:
            embeddings: Current node embeddings, shape (batch, num_nodes, embed_dim).
            state: Current environment state (TensorDict or compatible).

        Returns:
            torch.Tensor: Projected step context embedding, shape (batch, 1, embed_dim).
        """
        batch_size = embeddings.size(0)
        current_node = state.get_current_node() if hasattr(state, "get_current_node") else state.get("current_node")

        if current_node.dim() > 1:
            current_node = current_node.squeeze(-1)  # [B]

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
            unvisited_mask = (~visited).float()
            num_nodes = waste.shape[-1]
            unvisited_waste_mean = (waste * unvisited_mask).sum(dim=-1, keepdim=True).unsqueeze(1) / num_nodes
        else:
            unvisited_waste_mean = torch.zeros(batch_size, 1, 1, device=embeddings.device)

        # 3. Mean Euclidean distance from current node to unvisited profitable nodes [B, 1, 1]
        if locs is not None and visited is not None and waste is not None:
            cur_idx = current_node.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, 2)
            cur_pos = locs.gather(1, cur_idx).squeeze(1)  # [B, 2]
            dist_to_all = torch.norm(locs - cur_pos.unsqueeze(1), dim=-1)  # [B, N]
            unvisited_profitable = (~visited) & (waste > 0)
            unvisited_profitable[:, 0] = False  # exclude depot
            count = unvisited_profitable.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
            mean_dist = (dist_to_all * unvisited_profitable.float()).sum(dim=-1, keepdim=True) / count
            mean_dist_unvisited = mean_dist.unsqueeze(1)  # [B, 1, 1]
        else:
            mean_dist_unvisited = torch.zeros(batch_size, 1, 1, device=embeddings.device)

        # 4. Remaining vehicle capacity [B, 1, 1]
        remaining_cap = td.get("remaining_capacity", None)
        if remaining_cap is None:
            remaining_cap = td.get(
                "current_load",
                torch.zeros(batch_size, device=embeddings.device),
            )
        if remaining_cap.dim() == 1:
            remaining_cap = remaining_cap.unsqueeze(-1)
        if remaining_cap.dim() == 2:
            remaining_cap = remaining_cap.unsqueeze(1)  # [B, 1, 1]

        # Concat all signals and project [B, 1, embed_dim+3] → [B, 1, embed_dim]
        combined = torch.cat([current_node_embed, unvisited_waste_mean, mean_dist_unvisited, remaining_cap], dim=-1)
        return self.project_step_context(combined)

    @property
    def step_context_dim(self) -> int:
        """Gets the dimensionality of the fused MVPTP step context.

        Returns:
            int: ``embed_dim + MVPTP_STEP_CONTEXT_OFFSET`` (= embed_dim + 3).
        """
        return self.embed_dim + MVPTP_STEP_CONTEXT_OFFSET
