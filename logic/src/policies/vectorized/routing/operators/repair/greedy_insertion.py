"""Greedy insertion repair operator.

This module provides a GPU-accelerated implementation of the greedy insertion
repair heuristic, which sequentially reinserts nodes into the position that
results in the minimum cost increase.

Attributes:
    vectorized_greedy_insertion: Inserts nodes greedily into the first valid position that minimizes cost.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.repair.greedy_insertion import vectorized_greedy_insertion
    >>> tours, wastes = vectorized_greedy_insertion(tours, removed_nodes, dist_matrix)
"""

from __future__ import annotations

from typing import Optional

import torch


def vectorized_greedy_insertion(
    tours: torch.Tensor,
    removed_nodes: torch.Tensor,
    dist_matrix: torch.Tensor,
    wastes: Optional[torch.Tensor] = None,
    capacity: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized greedy insertion.

    Sequentially inserts each node from removed_nodes into the best position
    in tours based on the delta cost.

    Args:
        tours: Batch of current sequences of shape [B, N_curr].
        removed_nodes: Batch of nodes to insert of shape [B, N_rem].
        dist_matrix: Edge cost tensor of shape [B, N, N].
        wastes: Node demands of shape [B, N] or [N].
        capacity: Vehicle capacity limit.
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: Repaired tours of shape [B, N_curr + N_rem].
    """
    B, N_curr = tours.shape
    N_rem = removed_nodes.shape[1]

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)

    for i in range(N_rem):
        node_to_insert = removed_nodes[:, i : i + 1]

        # 1. Compute costs and find best position
        costs = _compute_greedy_insertion_costs(tours, node_to_insert, dist_matrix)
        _, best_indices = torch.min(costs, dim=1)
        insert_pos = best_indices.view(B, 1)

        # 2. Insert node
        tours = _apply_insertion(tours, node_to_insert, insert_pos)
        N_curr += 1

    return tours


def _compute_greedy_insertion_costs(tours: torch.Tensor, node: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """Computes insertion costs for a single node at all positions in the tour.

    Args:
        tours: Sequences of shape [B, N].
        node: Node to insert of shape [B, 1].
        dist: Distance matrix of shape [B, Nmax, Nmax].

    Returns:
        torch.Tensor: Delta cost tensor of shape [B, N].
    """
    B, N_curr = tours.shape
    device = tours.device

    t_prev = tours
    t_next = torch.roll(tours, -1, dims=1)
    node_exp = node.expand(-1, N_curr)

    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N_curr)
    d_pn = dist[b_idx, t_prev, node_exp]
    d_nn = dist[b_idx, node_exp, t_next]
    d_pn_exist = dist[b_idx, t_prev, t_next]

    return d_pn + d_nn - d_pn_exist


def _apply_insertion(tours: torch.Tensor, node: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Inserts a node into the tour at the specified position.

    Args:
        tours: Batch sequences of shape [B, N].
        node: Node to insert of shape [B, 1].
        pos: Insertion index of shape [B, 1].

    Returns:
        torch.Tensor: Updated sequences of shape [B, N+1].
    """
    B, N_curr = tours.shape
    device = tours.device

    new_tours = torch.zeros((B, N_curr + 1), dtype=tours.dtype, device=device)
    seq = torch.arange(N_curr, device=device).unsqueeze(0).expand(B, N_curr)

    mask_left = seq <= pos
    write_idx = torch.where(mask_left, seq, seq + 1)

    new_tours.scatter_(1, write_idx, tours)
    new_tours.scatter_(1, pos + 1, node)
    return new_tours
