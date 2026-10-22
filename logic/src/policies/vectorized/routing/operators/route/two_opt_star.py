"""2-opt* local search operator.

This module provides a GPU-accelerated implementation of the 2-opt* heuristic,
frequently referred to as a "Tail Swap", which improves multi-route solutions
by exchanging the terminal segments of two different routes.

Attributes:
    vectorized_two_opt_star: Exchanges the 'tail' segments of two routes to
        improve fleet-level efficiency.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.route.two_opt_star import vectorized_two_opt_star
    >>> optimized_tours = vectorized_two_opt_star(tours, dist_matrix)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_two_opt_star(
    tours: torch.Tensor,
    dist_matrix: torch.Tensor,
    max_iterations: int = 200,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized 2-opt* (Tail Swap) operator.

    Exchanges the 'tail' segments (nodes following positions i and j) between
    two distinct routes. This move is particularly effective for reconnecting
    routes and improving fleet-level efficiency.

    Args:
        tours: Batch of node sequences of shape [B, L].
        dist_matrix: Edge cost tensor of shape [B, N, N].
        max_iterations: Number of random tail-swap pairs to evaluate.
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: Updated batch of tours of shape [B, L].
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
        # 1. Sample indices i, j
        idx = torch.randint(1, max_len - 1, (B, 2), device=device, generator=generator)
        i, j = idx[:, 0:1], idx[:, 1:2]

        node_i, node_j = torch.gather(tours, 1, i), torch.gather(tours, 1, j)
        mask = (node_i != 0) & (node_j != 0) & (i != j)
        if not mask.any():
            continue

        # 2. Identify Routes
        end_i, end_j, route_mask = _identify_two_opt_star_routes(tours, i, j, seq, B)
        mask &= route_mask
        if not mask.any():
            continue

        # 3. Compute Gain
        gain = _compute_two_opt_star_gain(tours, dist_matrix, node_i, node_j, i, j, batch_indices)
        improved = (gain > IMPROVEMENT_EPSILON) & mask

        if improved.any():
            tours = _apply_two_opt_moves(tours, improved, i, j, end_i, end_j, max_len, seq, device)

    return tours


def _identify_two_opt_star_routes(
    tours: torch.Tensor, i: torch.Tensor, j: torch.Tensor, seq: torch.Tensor, B: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detects route boundaries for inter-route tail exchange.

    Args:
        tours: Batch of sequences of shape [B, N].
        i: First split position of shape [B, 1].
        j: Second split position of shape [B, 1].
        seq: Coordinate sequence of shape [B, N].
        B: Batch size.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - end_i: Calculated end index for the first identified route.
            - end_j: Calculated end index for the second identified route.
            - mask: Boolean validity mask for the inter-route exchange.
    """
    is_zero = tours == 0
    end_i = torch.argmax((is_zero & (seq > i)).float(), dim=1).view(B, 1)
    end_j = torch.argmax((is_zero & (seq > j)).float(), dim=1).view(B, 1)

    valid = (end_i > i) & (end_j > j)
    inter_route = end_i != end_j
    return end_i, end_j, valid & inter_route


def _compute_two_opt_star_gain(
    tours: torch.Tensor,
    dist: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    b_idx: torch.Tensor,
) -> torch.Tensor:
    """Calculates gain for breaking edges (u, un) and (v, vn).

    Args:
        tours: Batch of sequences of shape [B, N].
        dist: Distance matrix of shape [B, N+1, N+1].
        u: Node at position i of shape [B, 1].
        v: Node at position j of shape [B, 1].
        i: First position index of shape [B, 1].
        j: Second position index of shape [B, 1].
        b_idx: Batch indices of shape [B, 1].

    Returns:
        torch.Tensor: Gain tensor of shape [B, 1].
    """
    un = torch.gather(tours, 1, i + 1)
    vn = torch.gather(tours, 1, j + 1)
    d_curr = dist[b_idx, u, un] + dist[b_idx, v, vn]
    d_new = dist[b_idx, u, vn] + dist[b_idx, v, un]
    return d_curr - d_new


def _apply_two_opt_moves(
    tours: torch.Tensor,
    improved: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    end_i: torch.Tensor,
    end_j: torch.Tensor,
    max_len: int,
    seq: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Batch tail swap using conditional segment mapping.

    Args:
        tours: Batch of sequences of shape [B, N].
        improved: Improvement flag of shape [B].
        i: Position in first route of shape [B, 1].
        j: Position in second route of shape [B, 1].
        end_i: Finish index of first route of shape [B, 1].
        end_j: Finish index of second route of shape [B, 1].
        max_len: Total length of tour sequence.
        seq: Sequence indexes of shape [B, N].
        device: Hardware identification locator.

    Returns:
        torch.Tensor: Updated tours of shape [B, N].
    """
    B = tours.shape[0]
    idx_map = seq.clone()

    # R1 before R2
    m_i_lt_j = (end_i <= j) & improved.view(B, 1)
    if m_i_lt_j.any():
        idx_map = _map_tail_swap(idx_map, m_i_lt_j, i, j, end_i, end_j, B, max_len)

    # R2 before R1
    m_j_lt_i = (end_j <= i) & improved.view(B, 1)
    if m_j_lt_i.any():
        idx_map = _map_tail_swap(idx_map, m_j_lt_i, j, i, end_j, end_i, B, max_len)

    return torch.gather(tours, 1, idx_map)


def _map_tail_swap(
    idx_map: torch.Tensor,
    mask: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    end_i: torch.Tensor,
    end_j: torch.Tensor,
    B: int,
    max_len: int,
) -> torch.Tensor:
    """Constructs the index mapping for asymmetric route ordering.

    Args:
        idx_map: Current index map of shape [B, N].
        mask: Activation mask for this swap configuration of shape [B, N].
        i: Source route split point of shape [B, 1].
        j: Target route split point of shape [B, 1].
        end_i: Source route end of shape [B, 1].
        end_j: Target route end of shape [B, 1].
        B: Batch size.
        max_len: Total tour length.

    Returns:
        torch.Tensor: Modified index map of shape [B, N].
    """
    seq_b = torch.arange(max_len, device=idx_map.device).view(1, max_len).expand(B, max_len)

    len_t2 = end_j - (j + 1)
    len_gap = j - end_i + 1

    # New R1 tail: maps to [j+1, j+len_t2]
    m1 = mask & (seq_b > i) & (seq_b <= i + len_t2)
    idx_map[m1] = (j + 1).view(B, 1).expand(-1, max_len)[m1] + (seq_b[m1] - (i + 1).view(B, 1).expand(-1, max_len)[m1])

    # Gap shift: maps to [end_i, j]
    m_gap = mask & (seq_b > i + len_t2) & (seq_b <= i + len_t2 + len_gap)
    idx_map[m_gap] = (end_i).view(B, 1).expand(-1, max_len)[m_gap] + (
        seq_b[m_gap] - (i + len_t2 + 1).view(B, 1).expand(-1, max_len)[m_gap]
    )

    # New R2 tail: maps to [i+1, end_i-1]
    m2 = mask & (seq_b > i + len_t2 + len_gap) & (seq_b < end_j.view(B, 1))
    idx_map[m2] = (i + 1).view(B, 1).expand(-1, max_len)[m2] + (
        seq_b[m2] - (i + len_t2 + len_gap + 1).view(B, 1).expand(-1, max_len)[m2]
    )

    return idx_map
