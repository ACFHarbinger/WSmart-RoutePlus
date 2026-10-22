"""Worst removal operator.

This module provides a GPU-accelerated implementation of the worst removal
heuristic, which greedily identifies and ejects nodes from the tour that
contribute most to its total distance.

Attributes:
    vectorized_worst_removal: Ejects nodes that contribute most to total tour distance.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.destroy.worst_removal import vectorized_worst_removal
    >>> tours, removed_nodes = vectorized_worst_removal(tours, dist_matrix, n_remove)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def vectorized_worst_removal(
    tours: torch.Tensor,
    dist_matrix: torch.Tensor,
    n_remove: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized worst removal (highest cost-savings ejection).

    Calculates the savings (edge cost reduction) obtained by removing each
    node and its adjacent edges. Prioritizes nodes with the highest savings.

    Args:
        tours: Batch of node sequences of shape [B, N].
        dist_matrix: Global distance matrix of shape [B, N_all, N_all] or [1, N_all, N_all].
        n_remove: Number of nodes to remove per tour.
        generator: PyTorch generator for random number generation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - new_tours: Sequence after removal of shape [B, N - n_remove].
            - removed_nodes: IDs of the ejected nodes of shape [B, n_remove].
    """
    B, N = tours.shape
    device = tours.device

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    # Shifted tours
    tours_prev = torch.roll(tours, 1, dims=1)
    tours_next = torch.roll(tours, -1, dims=1)

    # Helper to gather distances
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)

    d_prev_node = dist_matrix[batch_idx, tours_prev, tours]
    d_node_next = dist_matrix[batch_idx, tours, tours_next]
    d_prev_next = dist_matrix[batch_idx, tours_prev, tours_next]

    savings = d_prev_node + d_node_next - d_prev_next

    # Mask out depots/padding (should not remove them)
    customers_mask = tours > 0
    savings[~customers_mask] = -float("inf")

    # Pick top n_remove savings
    _, remove_indices = torch.topk(savings, k=n_remove, dim=1)

    removed_nodes = torch.gather(tours, 1, remove_indices)

    # Create mask and collapse
    remove_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    remove_mask.scatter_(1, remove_indices, True)

    keep_mask = ~remove_mask
    new_tours = tours[keep_mask].view(B, N - n_remove)

    return new_tours, removed_nodes
