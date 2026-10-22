"""Random removal operator.

This module provides a GPU-accelerated implementation of the random removal
heuristic, which stochastically selects nodes to eject from tours, providing
diversification for large neighborhood search.

Attributes:
    vectorized_random_removal: Randomly removes nodes from tours.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.destroy.random_removal import vectorized_random_removal
    >>> tours, removed_nodes = vectorized_random_removal(tours, n_remove)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def vectorized_random_removal(
    tours: torch.Tensor,
    n_remove: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized random removal of nodes from tours.

    Identifies valid customers (non-depots) and stochastically selects
    n_remove nodes per batch instance using random scoring.

    Args:
        tours: Batch of node sequences of shape [B, N].
        n_remove: Number of nodes to eject from each tour.
        generator: PyTorch generator for random number generation.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - compressed_tours: Sequence after removal of shape [B, N - n_remove].
            - removed_nodes: IDs of the ejected nodes of shape [B, n_remove].
    """
    B, N = tours.shape
    device = tours.device

    # Identify valid customers (non-zero)
    customers_mask = tours > 0

    # Create random scores for sorting
    scores = torch.rand((B, N), device=device, generator=generator)
    scores[~customers_mask] = -1.0  # Depots/Padding have low score

    # Get indices of top k scores
    _, remove_indices = torch.topk(scores, k=n_remove, dim=1)  # (B, n_remove)

    # Gather removed nodes
    removed_nodes = torch.gather(tours, 1, remove_indices)

    # Create mask and collapse
    remove_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    remove_mask.scatter_(1, remove_indices, True)

    keep_mask = ~remove_mask

    # We need to collapse the kept nodes.
    # Since we remove exactly n_remove nodes from each row,
    # the number of kept nodes is N - n_remove.
    new_tours = tours[keep_mask].view(B, N - n_remove)

    return new_tours, removed_nodes
