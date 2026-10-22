"""Swap* local search operator.

This module provides a GPU-accelerated implementation of the Swap* operator,
which performs inter-route exchanges of nodes and re-optimizes their insertion
points in the target routes for maximum efficiency.

Attributes:
    vectorized_swap_star: Inter-route node exchange with optimal re-insertion.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.route.swap_star import vectorized_swap_star
    >>> optimized_tours = vectorized_swap_star(tours, dist_matrix)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_swap_star(
    tours: torch.Tensor,
    dist_matrix: torch.Tensor,
    max_iterations: int = 100,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized Swap* operator for inter-route optimization.

    Exchanges nodes 'u' and 'v' between two different routes and finds the
    mathematically optimal insertion point for each in their new route.

    Args:
        tours: Batch of node sequences of shape [B, L].
        dist_matrix: Edge cost tensor of shape [B, N, N].
        max_iterations: Number of random node pairs to evaluate.
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: Optimized tours of shape [B, L].
    """
    device = tours.device
    B, max_len = tours.shape

    batch_indices = torch.arange(B, device=device).view(B, 1)
    seq = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        # 1. Sample u (i) and v (j)
        idx = torch.randint(1, max_len - 1, (B, 2), device=device, generator=generator)
        i, j = idx[:, 0:1], idx[:, 1:2]

        node_i, node_j = torch.gather(tours, 1, i), torch.gather(tours, 1, j)
        mask = (node_i != 0) & (node_j != 0) & (i != j)
        if not mask.any():
            continue

        # 2. Identify Routes
        start_i, end_i, start_j, end_j, route_mask = _identify_routes(tours, i, j, seq, B)
        mask &= route_mask
        if not mask.any():
            continue

        # 3. Compute Gains and Find Best Insertions
        total_gain, best_ins_u, best_ins_v = _compute_swap_star_gains(
            tours,
            dist_matrix,
            node_i,
            node_j,
            i,
            j,
            start_i,
            end_i,
            start_j,
            end_j,
            batch_indices,
            max_len,
            seq,
            device,
        )

        improved = (total_gain > IMPROVEMENT_EPSILON) & mask
        if improved.any():
            tours = _apply_swap_star_moves(tours, improved, i, j, best_ins_u, best_ins_v, max_len, device)

    return tours


def _identify_routes(
    tours: torch.Tensor, i: torch.Tensor, j: torch.Tensor, seq: torch.Tensor, B: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detects route start/end points relative to sampled indices.

    Args:
        tours: Batch of sequences of shape [B, N].
        i: First node position of shape [B, 1].
        j: Second node position of shape [B, 1].
        seq: Index sequence of shape [B, N].
        B: Batch size.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - start_i: Identified starting index of the first route.
            - end_i: Identified ending index of the first route.
            - start_j: Identified starting index of the second route.
            - end_j: Identified ending index of the second route.
            - mask: Boolean inter-route validity flag.
    """
    is_zero = tours == 0

    start_i = torch.max(torch.where(is_zero & (seq < i), seq, -1), dim=1)[0].view(B, 1)
    end_i = torch.argmax((is_zero & (seq > i)).float(), dim=1).view(B, 1)

    start_j = torch.max(torch.where(is_zero & (seq < j), seq, -1), dim=1)[0].view(B, 1)
    end_j = torch.argmax((is_zero & (seq > j)).float(), dim=1).view(B, 1)

    valid = (end_i > i) & (start_i < i) & (end_j > j) & (start_j < j) & (start_i >= 0) & (start_j >= 0)
    inter_route = start_i != start_j
    return start_i, end_i, start_j, end_j, valid & inter_route


def _compute_swap_star_gains(
    tours: torch.Tensor,
    dist: torch.Tensor,
    node_i: torch.Tensor,
    node_j: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    start_i: torch.Tensor,
    end_i: torch.Tensor,
    start_j: torch.Tensor,
    end_j: torch.Tensor,
    b_idx: torch.Tensor,
    max_len: int,
    seq: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluates differential insertion/removal costs for the swap pair.

    Args:
        tours: Batch of sequences of shape [B, N].
        dist: Distance matrix of shape [B, N, N].
        node_i: Node u identifier of shape [B, 1].
        node_j: Node v identifier of shape [B, 1].
        i: Position of u in current sequence of shape [B, 1].
        j: Position of v in current sequence of shape [B, 1].
        start_i: First route start index of shape [B, 1].
        end_i: First route end index of shape [B, 1].
        start_j: Second route start index of shape [B, 1].
        end_j: Second route end index of shape [B, 1].
        b_idx: Batch indices of shape [B, 1].
        max_len: Total tour length N.
        seq: Sequence index coordinates of shape [B, N].
        device: Hardware identification locator.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - total_gain: Combined cost reduction for the swap.
            - idx_u: Best new insertion position for node u.
            - idx_v: Best new insertion position for node v.
    """
    # Removal gains
    gain_i = (
        dist[b_idx, torch.gather(tours, 1, i - 1), node_i]
        + dist[b_idx, node_i, torch.gather(tours, 1, i + 1)]
        - dist[b_idx, torch.gather(tours, 1, i - 1), torch.gather(tours, 1, i + 1)]
    )
    gain_j = (
        dist[b_idx, torch.gather(tours, 1, j - 1), node_j]
        + dist[b_idx, node_j, torch.gather(tours, 1, j + 1)]
        - dist[b_idx, torch.gather(tours, 1, j - 1), torch.gather(tours, 1, j + 1)]
    )

    # Best insertions
    next_nodes = torch.roll(tours, shifts=-1, dims=1)
    b_rows = b_idx.expand(-1, max_len)

    # u into J
    cost_u = (
        dist[b_rows, tours, node_i.expand(-1, max_len)]
        + dist[b_rows, node_i.expand(-1, max_len), next_nodes]
        - dist[b_rows, tours, next_nodes]
    )
    mask_j = (seq >= start_j) & (seq < end_j) & (seq != j) & (seq != j - 1)
    val_u, idx_u = torch.min(torch.where(mask_j, cost_u, torch.tensor(float("inf"), device=device)), dim=1)

    # v into I
    cost_v = (
        dist[b_rows, tours, node_j.expand(-1, max_len)]
        + dist[b_rows, node_j.expand(-1, max_len), next_nodes]
        - dist[b_rows, tours, next_nodes]
    )
    mask_i = (seq >= start_i) & (seq < end_i) & (seq != i) & (seq != i - 1)
    val_v, idx_v = torch.min(torch.where(mask_i, cost_v, torch.tensor(float("inf"), device=device)), dim=1)

    return (
        gain_i + gain_j - val_u.view(-1, 1) - val_v.view(-1, 1),
        idx_u.view(-1, 1),
        idx_v.view(-1, 1),
    )


def _apply_swap_star_moves(
    tours: torch.Tensor,
    improved: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    ins_u: torch.Tensor,
    ins_v: torch.Tensor,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Updates sequence batch by repositioning u and v based on weights.

    Args:
        tours: Batch of sequences of shape [B, N].
        improved: Improvement flag of shape [B, 1].
        i: Old position of node u of shape [B, 1].
        j: Old position of node v of shape [B, 1].
        ins_u: New optimal position for node u of shape [B, 1].
        ins_v: New optimal position for node v of shape [B, 1].
        max_len: Total tour length N.
        device: Hardware identification locator.

    Returns:
        torch.Tensor: Modified tours of shape [B, N].
    """
    seq_b_flat = torch.arange(max_len, device=device).view(1, max_len).expand(tours.shape[0], max_len).float()
    weights = seq_b_flat * 10.0

    new_weights = weights.clone()
    new_weights.scatter_(1, i, ins_u.float() * 10.0 + 5.0)
    new_weights.scatter_(1, j, ins_v.float() * 10.0 + 5.1)

    weights = torch.where(improved, new_weights, weights)
    return torch.gather(tours, 1, torch.argsort(weights, dim=1))
