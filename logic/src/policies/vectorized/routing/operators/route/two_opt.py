"""2-opt local search operator.

This module provides a GPU-accelerated implementation of the 2-opt heuristic,
evaluating all possible edge-swap combinations in parallel across
problem batches using meshgrid-based gain computation.

Attributes:
    vectorized_two_opt: Vectorized 2-opt local search using parallel gain evaluation.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.route.two_opt import vectorized_two_opt
    >>> optimized_tours = vectorized_two_opt(tours, dist_matrix)
"""

from __future__ import annotations

from typing import Optional

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_two_opt(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    max_iterations: int = 200,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized 2-opt local search using parallel gain evaluation.

    Iteratively improves tours by reversing segments. For each pair of edges
    (i, i+1) and (j, j+1), it checks if swapping them reduces total distance.

    Args:
        tours: Batch of node sequences of shape [B, N].
        distance_matrix: Edge cost tensor of shape [B, N+1, N+1] or [N+1, N+1].
        max_iterations: Maximum number of improvement cycles.
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: Optimized tours of shape [B, N].
    """
    device = distance_matrix.device

    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)

    # Handle distance_matrix expansion
    if distance_matrix.dim() == 2:
        distance_matrix = distance_matrix.unsqueeze(0)

    B, N = tours.shape
    if N < 4:
        return tours if is_batch else tours.squeeze(0)

    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    batch_indices = torch.arange(B, device=device).view(B, 1)

    # Pre-generate all possible edge pairs (i, j)
    indices = torch.arange(N, device=device)
    i_orig = indices[1:-2]
    j_orig = indices[2:-1]
    I_grid, J_grid = torch.meshgrid(i_orig, j_orig, indexing="ij")
    pair_mask = J_grid > I_grid
    I_vals, J_vals = I_grid[pair_mask], J_grid[pair_mask]
    K = I_vals.size(0)

    for _ in range(max_iterations):
        # 1. Compute gains for all pairs
        gains = _compute_two_opt_gains(tours, distance_matrix, I_vals, J_vals, batch_indices, B, K)

        # 2. Find best gain
        best_gain, best_idx = torch.max(gains, dim=1)
        improved = best_gain > IMPROVEMENT_EPSILON
        if not improved.any():
            break

        # 3. Apply moves
        tours = _apply_two_opt_moves(tours, improved, I_vals, J_vals, best_idx, B, N, device)

    return tours if is_batch else tours.squeeze(0)


def _compute_two_opt_gains(
    tours: torch.Tensor,
    dist: torch.Tensor,
    I_vals: torch.Tensor,
    J_vals: torch.Tensor,
    b_idx: torch.Tensor,
    B: int,
    K: int,
) -> torch.Tensor:
    """Calculates edge weight differential for all candidate swaps.

    Args:
        tours: Batch of sequences of shape [B, N].
        dist: Distance matrix of shape [B, N+1, N+1].
        I_vals: Starting indices of first edge of shape [K].
        J_vals: Ending indices of second edge of shape [K].
        b_idx: Batch sequence index mapping of shape [B, 1].
        B: Batch size.
        K: Number of coordinate pairs.

    Returns:
        torch.Tensor: Gain tensor of shape [B, K].
    """
    t_prev_i = tours[:, I_vals - 1]
    t_curr_i = tours[:, I_vals]
    t_curr_j = tours[:, J_vals]
    t_next_j = tours[:, J_vals + 1]

    b_exp = b_idx.expand(B, K)
    d_curr = dist[b_exp, t_prev_i, t_curr_i] + dist[b_exp, t_curr_j, t_next_j]
    d_next = dist[b_exp, t_prev_i, t_curr_j] + dist[b_exp, t_curr_i, t_next_j]
    return d_curr - d_next


def _apply_two_opt_moves(
    tours: torch.Tensor,
    improved: torch.Tensor,
    I_vals: torch.Tensor,
    J_vals: torch.Tensor,
    best_idx: torch.Tensor,
    B: int,
    N: int,
    device: torch.device,
) -> torch.Tensor:
    """Batch reversal of segments using index mapping.

    Args:
        tours: Batch of sequences of shape [B, N].
        improved: Boolean improvement flag of shape [B].
        I_vals: Starting indices of first edge of shape [K].
        J_vals: Ending indices of second edge of shape [K].
        best_idx: Coordinates of max gain of shape [B].
        B: Batch size.
        N: Tour length.
        device: Hardware identification locator.

    Returns:
        torch.Tensor: Updated tours of shape [B, N].
    """
    target_i = I_vals[best_idx].view(B, 1)
    target_j = J_vals[best_idx].view(B, 1)

    k = torch.arange(N, device=device).view(1, N).expand(B, N)
    idx_map = torch.arange(N, device=device).view(1, N).expand(B, N).clone()

    reversal_mask = (k >= target_i) & (k <= target_j) & improved.view(B, 1)
    rev_idx = target_i + target_j - k
    idx_map[reversal_mask] = rev_idx[reversal_mask]

    return torch.gather(tours, 1, idx_map)
