"""3-opt local search operator.

This module provides a GPU-accelerated implementation of the 3-opt heuristic,
evaluating multiple reconnection ways (cases 4-7) for three-edge removals
in parallel across problem batches.

Attributes:
    vectorized_three_opt: Vectorized 3-opt local search using parallel sampling for efficiency.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.route.three_opt import vectorized_three_opt
    >>> tours = vectorized_three_opt(tours, dist_matrix, max_iterations)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_three_opt(
    tours: torch.Tensor,
    dist_matrix: torch.Tensor,
    max_iterations: int = 100,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized 3-opt local search using parallel sampling for efficiency.

    Randomly samples three edge break points and evaluates the gain for
    four distinct reconnection cases (Symmetrically different 3-opt moves).

    Args:
        tours: Batch of node sequences of shape [B, L].
        dist_matrix: Edge cost tensor of shape [B, N, N] or [N, N].
        max_iterations: Number of random sampling cycles to execute.
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: Optimized tours of shape [B, L].
    """
    device = tours.device
    B, max_len = tours.shape

    if max_len < 6:
        return tours

    batch_indices = torch.arange(B, device=device).view(B, 1)

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        # 1. Sample 3 indices i < j < k
        idx = torch.sort(
            torch.randint(1, max_len - 1, (B, 3), device=device, generator=generator),
            dim=1,
        ).values
        i, j, k = idx[:, 0:1], idx[:, 1:2], idx[:, 2:3]

        mask = (torch.gather(tours, 1, i) != 0) & (torch.gather(tours, 1, j) != 0) & (torch.gather(tours, 1, k) != 0)
        mask &= (j > i + 1) & (k > j + 1)
        if not mask.any():
            continue

        # 2. Compute gains
        best_gain, best_case = _compute_three_opt_gains(tours, dist_matrix, i, j, k, batch_indices)

        improved = (best_gain > IMPROVEMENT_EPSILON) & mask.squeeze(1)
        if improved.any():
            tours = _apply_three_opt_moves(tours, improved, best_case, i, j, k, max_len, device)

    return tours


def _compute_three_opt_gains(
    tours: torch.Tensor,
    dist: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    b_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes edge differential for cases 4-7 of 3-opt.

    Args:
        tours: Batch of sequences of shape [B, N].
        dist: Distance matrix of shape [B, N, N].
        i: First break point index of shape [B, 1].
        j: Second break point index of shape [B, 1].
        k: Third break point index of shape [B, 1].
        b_idx: Batch indices of shape [B, 1].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - best_gain: Maximum cost reduction found among cases.
            - best_case: Index of the winning reconnection strategy.
    """
    u, un = torch.gather(tours, 1, i), torch.gather(tours, 1, i + 1)
    v, vn = torch.gather(tours, 1, j), torch.gather(tours, 1, j + 1)
    w, wn = torch.gather(tours, 1, k), torch.gather(tours, 1, k + 1)

    d_base = dist[b_idx, u, un] + dist[b_idx, v, vn] + dist[b_idx, w, wn]

    g4 = d_base - (dist[b_idx, u, v] + dist[b_idx, un, w] + dist[b_idx, vn, wn])
    g5 = d_base - (dist[b_idx, u, vn] + dist[b_idx, w, un] + dist[b_idx, v, wn])
    g6 = d_base - (dist[b_idx, u, vn] + dist[b_idx, w, v] + dist[b_idx, un, wn])
    g7 = d_base - (dist[b_idx, u, w] + dist[b_idx, vn, un] + dist[b_idx, v, wn])

    return torch.max(torch.cat([g4, g5, g6, g7], dim=1), dim=1)


def _apply_three_opt_moves(
    tours: torch.Tensor,
    improved: torch.Tensor,
    best_case: torch.Tensor,
    i: torch.Tensor,
    j: torch.Tensor,
    k: torch.Tensor,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Batch recombination based on case IDs.

    Args:
        tours: Batch of tours of shape [B, N].
        improved: Improvement flag of shape [B].
        best_case: Case identifier of shape [B].
        i: First point of shape [B, 1].
        j: Second point of shape [B, 1].
        k: Third point of shape [B, 1].
        max_len: Total tour length N.
        device: Hardware identification locator.

    Returns:
        torch.Tensor: Modified tours of shape [B, N].
    """
    B = tours.shape[0]
    seq_b = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)
    idx_map = seq_b.clone()

    # Common segment lengths
    len3 = k - j

    # Case 4 (idx 0): S1, S2^R, S3^R, S4
    m0 = (best_case == 0) & improved
    idx_map = torch.where(m0.view(B, 1) & (seq_b > i) & (seq_b <= j), i + j + 1 - seq_b, idx_map)
    idx_map = torch.where(m0.view(B, 1) & (seq_b > j) & (seq_b <= k), j + k + 1 - seq_b, idx_map)

    # Case 5 (idx 1): S1, S3, S2, S4
    m1 = (best_case == 1) & improved
    idx_map = torch.where(m1.view(B, 1) & (seq_b > i) & (seq_b <= i + len3), seq_b + (j - i), idx_map)
    idx_map = torch.where(m1.view(B, 1) & (seq_b > i + len3) & (seq_b <= k), seq_b - (k - j), idx_map)

    # Case 6 (idx 2): S1, S3, S2^R, S4
    m2 = (best_case == 2) & improved
    idx_map = torch.where(m2.view(B, 1) & (seq_b > i) & (seq_b <= i + len3), seq_b + (j - i), idx_map)
    idx_map = torch.where(
        m2.view(B, 1) & (seq_b > i + len3) & (seq_b <= k),
        j - (seq_b - (i + len3 + 1)),
        idx_map,
    )

    # Case 7 (idx 3): S1, S3^R, S2, S4
    m3 = (best_case == 3) & improved
    idx_map = torch.where(m3.view(B, 1) & (seq_b > i) & (seq_b <= i + len3), k - (seq_b - (i + 1)), idx_map)
    idx_map = torch.where(m3.view(B, 1) & (seq_b > i + len3) & (seq_b <= k), seq_b - (k - j), idx_map)

    return torch.gather(tours, 1, idx_map)
