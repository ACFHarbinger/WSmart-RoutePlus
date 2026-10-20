"""Evaluation metrics for HGS.

Provides GPU-accelerated diversity metrics and fitness evaluation components
used to maintain population quality and exploration depth.

Attributes:
    calc_broken_pairs_distance: Diversity metric based on edge preservation.

Example:
    >>> diversity = calc_broken_pairs_distance(population)
"""

from __future__ import annotations

import torch


def calc_broken_pairs_distance(population: torch.Tensor) -> torch.Tensor:
    """Computes average Broken Pairs Distance (BPD) for each individual.

    BPD measures the diversity between solutions by identifying edges in one
    solution that are absent in another.
    Distance(A, B) = 1 - (|Edges(A) ∩ Edges(B)| / N)

    Args:
        population: Tensor of giant tours of shape [B, P, N], where P is the
            population size and N is the number of nodes.

    Returns:
        torch.Tensor: Normalized diversity scores of shape [B, P].
            Higher scores indicate greater distance from the pool.
    """
    B, P, N = population.size()
    device = population.device

    # 1. Construct Edge Hashes
    # Edges: (i, i+1) and (N-1, 0)
    # Create cyclic view
    next_nodes = torch.roll(population, shifts=-1, dims=2)

    # Sort u,v to be direction agnostic
    u = torch.min(population, next_nodes)
    v = torch.max(population, next_nodes)

    # Hash
    hashes = u * (N + 100) + v  # (B, P, N)

    # 2. Compute Pairwise Distances
    # expand for broadcast: match matrix memory usage is roughly B*P*P*N bytes.
    # We use a sequential loop over P to remain memory-efficient on smaller GPUs.
    intersections = torch.zeros((B, P, P), device=device)
    for i in range(P):
        target = hashes[:, i : i + 1, :]
        matches = hashes.unsqueeze(3) == target.unsqueeze(2)
        num_shared = matches.any(dim=3).sum(dim=2)  # (B, P)
        intersections[:, i, :] = num_shared

    # Distance = 1 - (intersection / N)
    dists = 1.0 - (intersections.float() / N)

    # Diversity of i = mean distance to others
    diversity = dists.sum(dim=2) / max(1, P - 1)
    return diversity
