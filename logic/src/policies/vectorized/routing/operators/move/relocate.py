"""Relocate local search operator.

This module provides a GPU-accelerated implementation of the Relocate operator,
which improves tours by moving a node from its current position and
re-inserting it at a more cost-effective location.

Attributes:
    vectorized_relocate: Relocates a node within a tour to minimize distance.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.move.relocate import vectorized_relocate
    >>> optimized_tours = vectorized_relocate(tours, dist_matrix)
"""

from __future__ import annotations

from typing import Optional

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_relocate(
    tours: torch.Tensor,
    dist_matrix: torch.Tensor,
    max_iterations: int = 200,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized Relocate operator using parallel sampling.

    Removes a sampler node 'i' and re-inserts it at position 'j' across
    the entire batch, applying the move only if it yields a positive gain.

    Args:
        tours: Batch of node sequences of shape [B, L].
        dist_matrix: Edge cost tensor of shape [B, N, N].
        max_iterations: Number of random relocation pairs to evaluate.
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: Optimized tours of shape [B, L].
    """
    device = tours.device
    B, max_len = tours.shape

    batch_indices = torch.arange(B, device=device).view(B, 1)

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        # 1. Sample indices i (to move) and j (insert after)
        idx = torch.randint(1, max_len - 1, (B, 2), device=device, generator=generator)
        i, j = idx[:, 0:1], idx[:, 1:2]

        node_i = torch.gather(tours, 1, i)
        mask = (node_i != 0) & (i != j) & (i != j + 1)
        if not mask.any():
            continue

        # 2. Compute gain
        gain = _compute_relocate_gain(tours, dist_matrix, node_i, i, j, batch_indices)
        improved = (gain > IMPROVEMENT_EPSILON) & mask

        if improved.any():
            # 3. Apply moves
            tours = _apply_relocate_move(tours, improved, i, j, max_len, device)

    return tours


def _compute_relocate_gain(
    tours: torch.Tensor,
    dist: torch.Tensor,
    node_i: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    b_idx: torch.Tensor,
) -> torch.Tensor:
    """Calculates improvement gain for relocating node V_i after node V_j.

    Args:
        tours: Batch of sequences of shape [B, L].
        dist: Batch of distance matrices of shape [B, N, N].
        node_i: Column vector of node IDs being moved of shape [B, 1].
        i: Current positions of nodes to move of shape [B, 1].
        j: Target insertion positions of shape [B, 1].
        b_idx: Batch indices for gathering of shape [B, 1].

    Returns:
        torch.Tensor: Calculated cost improvement per sample.
    """
    prev_i = torch.gather(tours, 1, i - 1)
    next_i = torch.gather(tours, 1, i + 1)
    node_j = torch.gather(tours, 1, j)
    next_j = torch.gather(tours, 1, j + 1)

    # Removal cost change
    d_remove = -dist[b_idx, prev_i, node_i] - dist[b_idx, node_i, next_i] + dist[b_idx, prev_i, next_i]
    # Insertion cost change
    d_insert = -dist[b_idx, node_j, next_j] + dist[b_idx, node_j, node_i] + dist[b_idx, node_i, next_j]
    return -(d_remove + d_insert)


def _apply_relocate_move(
    tours: torch.Tensor,
    improved: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Updates tour segment order using conditional index shifting.

    Args:
        tours: Input tour sequences of shape [B, L].
        improved: Activation mask for improving moves of shape [B, 1].
        i: IDs of moving nodes.
        j: Targeted insertion positions.
        max_len: Sequence length L.
        device: Hardware identification locator.

    Returns:
        torch.Tensor: Physically updated sequence batch.
    """
    B = tours.shape[0]
    seq_b = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)
    idx_map = seq_b.clone()

    # Case 1: i < j. Node i moves right. Indices in [i, j) shift left.
    m1 = (i < j) & improved
    shift_left = m1 & (seq_b >= i) & (seq_b < j)
    idx_map[shift_left] = seq_b[shift_left] + 1
    idx_map[m1 & (seq_b == j)] = i.expand(-1, max_len)[m1 & (seq_b == j)]

    # Case 2: j < i. Node i moves left. Indices in (j, i) shift right.
    m2 = (j < i) & improved
    shift_right = m2 & (seq_b > j) & (seq_b < i)
    idx_map[shift_right] = seq_b[shift_right] - 1
    idx_map[m2 & (seq_b == j + 1)] = i.expand(-1, max_len)[m2 & (seq_b == j + 1)]

    return torch.gather(tours, 1, idx_map)
