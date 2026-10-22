"""Type IV unstringing operator.

This module provides a GPU-accelerated implementation of the Type IV unstringing
heuristic, which improves tour quality by identifying a node to "unstring"
and reinserting it into a new position while reversing intermediate segments.

Attributes:
    vectorized_type_iv_unstringing: Vectorized Type IV Unstringing local search.

Example:
    >>> tours = torch.tensor([[0, 1, 2, 3, 4, 5, 0]])
    >>> dist = torch.randn(1, 6, 6)
    >>> opt_tours = vectorized_type_iv_unstringing(tours, dist)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_type_iv_unstringing(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    max_iterations: int = 50,
    sample_size: int = 30,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized Type IV Unstringing local search across a batch of tours.

    Identifies nodes to remove and reinsert in a 5-edge reconnection pattern
    that reverses segments to minimize total distance.

    Args:
        tours: Batch of node sequences of shape [B, N].
        distance_matrix: Pairwise node distances of shape [B, N, N] or [N, N].
        max_iterations: Maximum number of improvement passes.
        sample_size: Number of (j, l, k) tuples to evaluate per node.
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: Optimized batch of tours of shape [B, N].
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
    if N < 9:
        return tours if is_batch else tours.squeeze(0)

    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        improved_any = False
        for b in range(B):
            tour = tours[b]
            dist = distance_matrix[b]

            valid_indices = torch.where((tour > 0) & (tour < N))[0]
            if len(valid_indices) < 7:
                continue

            best_delta, best_move = _find_best_type_iv_move(tour, dist, valid_indices, sample_size, device, generator)

            if best_move is not None:
                i, j, l, k = best_move
                tours[b] = _apply_type_iv_move(tour, i, j, l, k)
                improved_any = True

        if not improved_any:
            break

    return tours if is_batch else tours.squeeze(0)


def _find_best_type_iv_move(
    tour: torch.Tensor,
    dist: torch.Tensor,
    valid_indices: torch.Tensor,
    sample_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    """Finds the best Type IV unstringing move for a single tour.

    Args:
        tour: Sequence of shape [N].
        dist: Cost matrix of shape [N, N].
        valid_indices: Nodes available for move of shape [V].
        sample_size: Search depth (number of randomized tuples).
        device: Execution device hardware placement.
        generator: RNG provider.

    Returns:
        Tuple[float, Optional[Tuple[int, int, int, int]]]: (best delta, move parameters).
    """
    N = len(tour)
    best_delta = 0.0
    best_move = None
    n_valid = len(valid_indices)

    for i_idx in range(n_valid):
        i = int(valid_indices[i_idx].item())

        # Sample or enumerate (j, l, k)
        if sample_size > 0:
            ns = min(sample_size, n_valid**3)
            j_s, l_s, k_s = (
                torch.randint(0, n_valid, (ns,), device=device, generator=generator),
                torch.randint(0, n_valid, (ns,), device=device, generator=generator),
                torch.randint(0, n_valid, (ns,), device=device, generator=generator),
            )
            # Valid Type IV: i < j < l < k
            valid = (j_s > i_idx) & (l_s > j_s) & (k_s > l_s)
            j_s, l_s, k_s = j_s[valid], l_s[valid], k_s[valid]
        else:
            continue

        for js, ls, ks in zip(j_s, l_s, k_s, strict=False):
            j, l, k = (
                int(valid_indices[js].item()),
                int(valid_indices[ls].item()),
                int(valid_indices[ks].item()),
            )
            delta = _evaluate_type_iv_move(tour, dist, i, j, l, k, N)
            if delta < best_delta - IMPROVEMENT_EPSILON:
                best_delta, best_move = delta, (i, j, l, k)

    return best_delta, best_move


def _evaluate_type_iv_move(tour: torch.Tensor, dist: torch.Tensor, i: int, j: int, l: int, k: int, N: int) -> float:
    """Calculates the delta cost for a specific Type IV move.

    Args:
        tour: Node sequence identifiers.
        dist: Pairwise distance weights.
        i: Target position for the unstrung node.
        j: First break point segment split.
        l: Second break point segment split.
        k: Third break point segment split.
        N: Tour length metadata.

    Returns:
        float: Total tour cost change.
    """
    v_ip, v_i, v_in = (
        int(tour[i - 1 if i > 0 else N - 1].item()),
        int(tour[i].item()),
        int(tour[(i + 1) % N].item()),
    )
    v_jp, v_j = int(tour[j - 1 if j > 0 else N - 1].item()), int(tour[j].item())
    v_lp, v_l = int(tour[l - 1 if l > 0 else N - 1].item()), int(tour[l].item())
    v_k, v_kn = int(tour[k].item()), int(tour[(k + 1) % N].item())

    # Pattern IV removal of node Vi
    rem = dist[v_ip, v_i] + dist[v_i, v_in] + dist[v_jp, v_j] + dist[v_lp, v_l] + dist[v_k, v_kn]
    ins = dist[v_ip, v_j] + dist[v_lp, v_in] + dist[v_k, v_jp] + dist[v_l, v_kn]
    return float((ins - rem).item())


def _apply_type_iv_move(tour: torch.Tensor, i: int, j: int, l: int, k: int) -> torch.Tensor:
    """Applies a Type IV unstringing move to the tour.

    Args:
        tour: Original node sequences.
        i: Position of the node to move.
        j: First segment break coordinate.
        l: Second segment break coordinate.
        k: Third segment break coordinate.

    Returns:
        torch.Tensor: Modified tour after segment reversals and re-insertion.
    """
    n = len(tour)
    tl = tour.tolist()

    i_prev, i_next = (i - 1) if i > 0 else (n - 1), (i + 1) % n
    _j_prev, _l_prev, k_next = (
        (j - 1) if j > 0 else (n - 1),
        (l - 1) if l > 0 else (n - 1),
        (k + 1) % n,
    )

    # Segments
    sc = tl[i_next:j] if i_next <= j - 1 else tl[i_next:] + tl[:j]
    sd = tl[l : k + 1] if l <= k else tl[l:] + tl[: k + 1]
    sa = tl[j:l] if j <= l - 1 else tl[j:] + tl[:l]
    # Remainder
    rem = tl[k_next : i_prev + 1] if k_next <= i_prev else tl[k_next:] + tl[: i_prev + 1]

    # Pattern: v_{i-1} -> sc -> sd -> sa_rev -> rem
    new_tour = [tl[i_prev]] + sc + sd + sa[::-1] + rem

    if 0 in new_tour:
        d_idx = new_tour.index(0)
        new_tour = new_tour[d_idx:] + new_tour[:d_idx]

    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
