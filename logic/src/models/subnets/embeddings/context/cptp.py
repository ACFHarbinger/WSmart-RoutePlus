"""CPTP-specific context embedding module.

This module provides ``CPTPContextEmbedder``, a capacity-aware extension of
``PTPContextEmbedder`` for the Capacitated Profitable Tour Problem (CPTP).

While the uncapacitated PTP context uses (unvisited waste, mean travel distance)
to guide early termination, the CPTP additionally exposes the vehicle's remaining
capacity so the decoder can learn to stop visiting bins when the vehicle is nearly
full even if profitable nodes remain.

Attributes:
    CPTPContextEmbedder: Context embedder for CPTP / orienteering with capacity.

Example:
    >>> from logic.src.models.subnets.embeddings.context.cptp import CPTPContextEmbedder
    >>> embedder = CPTPContextEmbedder(embed_dim=128)
    >>> context = embedder(embeddings, state)   # [B, 1, 128]
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from logic.src.constants.models import CPTP_STEP_CONTEXT_OFFSET, NODE_DIM

from .ptp import PTPContextEmbedder


class CPTPContextEmbedder(PTPContextEmbedder):
    """Context embedder for Capacitated Profitable Tour Problem (CPTP).

    Extends PTPContextEmbedder with a vehicle remaining-capacity scalar,
    giving the decoder three dynamic signals per decoding step:

    1. **Unvisited waste sum** — total profit still available on the map.
    2. **Mean distance to unvisited profitable nodes** — spatial cost signal.
    3. **Remaining capacity** — hard constraint signal.

    Attributes:
        init_embed (nn.Linear): Projection for non-depot customer features.
        init_embed_depot (nn.Linear): Projection for depot location.
        project_step_context (nn.Linear): Projection for fused step context.
    """

    def __init__(self, embed_dim: int, node_dim: int = NODE_DIM, temporal_horizon: int = 0) -> None:
        """Initializes CPTPContextEmbedder.

        Args:
            embed_dim: Target embedding dimensionality.
            node_dim: Base dimensionality of node features.
            temporal_horizon: Size of temporal feature window (if any).
        """
        super().__init__(embed_dim, node_dim, temporal_horizon)
        # Override project_step_context for CPTP's step_context_dim (embed_dim + 3)
        self.project_step_context = nn.Linear(self.step_context_dim, embed_dim)

    @property
    def step_context_dim(self) -> int:
        """Gets the dimensionality of the fused CPTP step context.

        Returns:
            int: ``embed_dim + CPTP_STEP_CONTEXT_OFFSET`` (= embed_dim + 3).
        """
        return self.embed_dim + CPTP_STEP_CONTEXT_OFFSET

    def _step_context(self, embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """Extracts current CPTP step context.

        Fuses four signals at each decoding step:
        1. Current node embedding.
        2. Sum of waste on unvisited nodes (remaining profit signal).
        3. Mean Euclidean distance to unvisited profitable nodes (travel cost signal).
        4. Remaining vehicle capacity (hard constraint signal).

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

        if embeddings.size(0) != batch_size:
            if embeddings.size(0) == 1:
                embeddings = embeddings.expand(batch_size, -1, -1)
            elif batch_size % embeddings.size(0) == 0:
                embeddings = embeddings.repeat_interleave(batch_size // embeddings.size(0), dim=0)
            else:
                embeddings = embeddings.expand(batch_size, -1, -1)

        # 1. Current node embedding [B, 1, embed_dim]
        current_node_embed = embeddings.gather(
            1,
            current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim),
        )

        # Gather shared state tensors
        td = cast(Any, getattr(state, "td", state))
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
            # This is a learned coordinate-space feature. Objective road matrices
            # are in kilometers and must not replace the checkpoint's input units.
            cur_idx = current_node.unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, locs.size(-1))
            cur_pos = locs.gather(1, cur_idx).squeeze(1)
            dist_to_all = torch.norm(locs - cur_pos.unsqueeze(1), dim=-1)
            unvisited_profitable = (~visited) & (waste > 0)
            unvisited_profitable[:, 0] = False
            count = unvisited_profitable.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
            mean_dist = (dist_to_all * unvisited_profitable.float()).sum(dim=-1, keepdim=True) / count
            mean_dist_unvisited = mean_dist.unsqueeze(1)
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


__all__ = ["CPTPContextEmbedder"]
