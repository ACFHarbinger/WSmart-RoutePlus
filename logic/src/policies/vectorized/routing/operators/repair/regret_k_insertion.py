"""Regret-K insertion repair operator.

This module provides a GPU-accelerated implementation of the Regret-K insertion
repair heuristic, which prioritizes reinserting nodes that have a large
difference (regret) between their best and kth-best insertion positions.

Attributes:
    vectorized_regret_k_insertion: Inserts nodes by choosing the position that maximizes "regret" (cost difference between best and k-th best insertions).

Example:
    >>> from logic.src.policies.vectorized.routing.operators.repair.regret_k_insertion import vectorized_regret_k_insertion
    >>> tours, wastes = vectorized_regret_k_insertion(tours, removed_nodes, dist_matrix)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def vectorized_regret_k_insertion(
    tours: torch.Tensor,
    removed_nodes: torch.Tensor,
    dist_matrix: torch.Tensor,
    k: int = 2,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized Regret-K insertion.

    Inserts removed nodes into the best positions based on the regret
    criterion: selecting the node that would result in the greatest lost
    opportunity if not inserted in its primary best position.

    Args:
        tours: Batch of current sequences of shape [B, N_curr].
        removed_nodes: Batch of nodes to insert of shape [B, N_rem].
        dist_matrix: Edge cost tensor of shape [B, N, N].
        k: Regret factor (kth-best minus best).
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: Repaired tours of shape [B, N_curr + N_rem].
    """
    B, _ = tours.shape
    device = tours.device
    N_rem = removed_nodes.shape[1]

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)

    pending_mask = torch.ones((B, N_rem), dtype=torch.bool, device=device)

    for _ in range(N_rem):
        # 1. Compute costs and regrets
        costs = _compute_insertion_costs(tours, removed_nodes, dist_matrix)
        regret, top_pos = _compute_regrets(costs, k, pending_mask)

        # 2. Select node with Max Regret
        _, node_idx_in_rem = torch.max(regret, dim=1)

        # 3. Apply Insertion
        node_to_insert = torch.gather(removed_nodes, 1, node_idx_in_rem.unsqueeze(1))
        insert_pos = torch.gather(top_pos, 1, node_idx_in_rem.unsqueeze(1))

        tours = _apply_insertion(tours, node_to_insert, insert_pos)
        pending_mask.scatter_(1, node_idx_in_rem.unsqueeze(1), False)

        if not pending_mask.any():
            break

    return tours


def _compute_insertion_costs(tours: torch.Tensor, removed_nodes: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    """Computes insertion costs for all pending nodes at all positions.

    Args:
        tours: Sequences of shape [B, N].
        removed_nodes: Available nodes of shape [B, N_rem].
        dist: Cost matrix of shape [B, Nmax, Nmax].

    Returns:
        torch.Tensor: 3D cost tensor of shape [B, N_rem, N].
    """
    B, N_curr = tours.shape
    _, N_rem = removed_nodes.shape
    device = tours.device

    t_prev = tours.unsqueeze(1).expand(-1, N_rem, -1)
    t_next = torch.roll(tours, -1, dims=1).unsqueeze(1).expand(-1, N_rem, -1)
    nodes = removed_nodes.unsqueeze(2).expand(-1, -1, N_curr)

    b_idx = torch.arange(B, device=device).view(B, 1, 1)
    d_pn = dist[b_idx, t_prev, nodes]
    d_nn = dist[b_idx, nodes, t_next]
    d_pn_exist = dist[b_idx, t_prev, t_next]

    return d_pn + d_nn - d_pn_exist


def _compute_regrets(costs: torch.Tensor, k: int, pending_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates regret based on top-k insertion costs.

    Args:
        costs: Price tensor of shape [B, N_rem, N].
        k: Regret depth.
        pending_mask: Availability tracker of shape [B, N_rem].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - regret_scores: Scores per node ([B, N_rem]).
            - best_positions: Best insertion indices ([B, N_rem]).
    """
    topk_vals, topk_indices = torch.topk(costs, k=k, dim=2, largest=False)

    best_costs = topk_vals[:, :, 0]
    kth_costs = topk_vals[:, :, -1]

    regret = kth_costs - best_costs
    regret[~pending_mask] = -float("inf")
    return regret, topk_indices[:, :, 0]


def _apply_insertion(tours: torch.Tensor, node: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Inserts a node into the tour at the specified position.

    Args:
        tours: Sequences of shape [B, N].
        node: Node to insert of shape [B, 1].
        pos: Target index of shape [B, 1].

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
