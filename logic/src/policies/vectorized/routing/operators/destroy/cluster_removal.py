"""Cluster removal operator.

This module provides a GPU-accelerated implementation of the cluster removal
heuristic, which identifies a spatial neighborhood around a seed node and
removes nodes within that cluster to facilitate large-scale local search.

Attributes:
    vectorized_cluster_removal: Removes a cluster of spatially related nodes from the tours.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.destroy.cluster_removal import vectorized_cluster_removal
    >>> tours, removed_nodes = vectorized_cluster_removal(tours, dist_matrix, n_remove)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def vectorized_cluster_removal(
    tours: torch.Tensor,
    dist_matrix: torch.Tensor,
    n_remove: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Removes a cluster of spatially related nodes from the tours.

    Selects a random seed node for each batch instance and removes the
    n_remove nearest neighbors according to the distance matrix.

    Args:
        tours: Batch of node sequences of shape [B, N].
        dist_matrix: Pairwise node distances of shape [B, Nmax, Nmax] or [Nmax, Nmax].
        n_remove: Number of nodes to eject from each tour.
        generator: Torch device-side RNG.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - compressed_tours: Sequence after removal of shape [B, N - n_remove].
            - removed_nodes: IDs of the ejected nodes of shape [B, n_remove].
    """
    B, N = tours.size()
    device = tours.device

    # 1. Pick a random seed node for each batch
    seed_idx = torch.randint(0, N, (B,), device=device, generator=generator)
    seed_nodes = torch.gather(tours, 1, seed_idx.unsqueeze(1)).squeeze(1)

    # 2. Find distances from seed nodes to all other nodes
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
    tour_distances = dist_matrix[batch_indices, seed_nodes.unsqueeze(1), tours]

    # 3. Take n_remove nearest nodes
    _, remove_idx = torch.topk(tour_distances, n_remove, dim=1, largest=False)

    # 4. Create mask and collapse
    mask = torch.ones_like(tours, dtype=torch.bool)
    mask[batch_indices[:, :n_remove], remove_idx] = False

    removed_nodes = torch.gather(tours, 1, remove_idx)

    # Compress tours (remove marked nodes)
    new_tours = tours[mask].view(B, N - n_remove)

    return new_tours, removed_nodes
