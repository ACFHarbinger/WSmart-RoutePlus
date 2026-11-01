"""CMPTPTWTC Embedding module.

This module provides the CMPTPTWTCInitEmbedding layer, which encodes locations, waste levels
and time windows for the Capacitated Multi-vehicle Profitable Tour Problem with Time Windows and Time
Constraints.

Attributes:
    CMPTPTWTCInitEmbedding: Initial feature encoder for CMPTPTWTC instances.

Example:
    >>> from logic.src.models.subnets.embeddings.cmptptwtc import CMPTPTWTCInitEmbedding
    >>> embed = CMPTPTWTCInitEmbedding(embed_dim=128)
    >>> h = embed(td)
"""

from __future__ import annotations

from typing import Any

import torch

from logic.src.models.subnets.embeddings.cmptp import CMPTPInitEmbedding


class CMPTPTWTCInitEmbedding(CMPTPInitEmbedding):
    """Initial embedding for CMPTPTWTC: CMPTP's node features plus each node's time window.

    Node features are x, y, waste, and the window's opening and closing times as fractions of
    the working day (``tw_early / day_hours``, ``tw_late / day_hours``). The depot's window
    is the whole day.
    """

    def __init__(self, embed_dim: int = 128, node_dim: int = 5, **kwargs: Any) -> None:
        """Initializes the embedding.

        Args:
            embed_dim: Output embedding dimension.
            node_dim: Per-node feature count (x, y, waste, window open, window close).
            kwargs: Forwarded to PTPInitEmbedding (temporal_horizon, legacy_depot_projection).
        """
        super().__init__(embed_dim=embed_dim, node_dim=node_dim, **kwargs)

    def _extra_node_features(self, td: Any, locs: torch.Tensor, waste: torch.Tensor) -> list:
        batch = locs.size(0)
        day = td.get("day_hours") if hasattr(td, "get") else None
        day = torch.as_tensor(10.0 if day is None else day, dtype=locs.dtype, device=locs.device).reshape(-1, 1)
        day = day.expand(batch, 1)
        early = td.get("tw_early") if hasattr(td, "get") else None
        late = td.get("tw_late") if hasattr(td, "get") else None
        n = waste.size(1)
        if early is None or late is None:
            early = torch.zeros(batch, n, device=locs.device)
            late = day.expand(batch, n)
        if early.size(1) == n - 1:  # bins only: the depot is open all day
            early = torch.cat([torch.zeros(batch, 1, device=locs.device), early], dim=1)
            late = torch.cat([day, late], dim=1)
        return [(early / day).unsqueeze(-1), (late / day).unsqueeze(-1)]


__all__ = ["CMPTPTWTCInitEmbedding"]
