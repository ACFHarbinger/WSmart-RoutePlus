"""Type III unstringing operator.

This module provides a GPU-accelerated implementation of the Type III unstringing
heuristic, which improves tour quality by identifying a node to "unstring"
and reinserting it into a new position while reversing intermediate segments.

Attributes:
    vectorized_type_iii_unstringing: Vectorized Type III Unstringing local search.

Example:
    >>> tours = torch.tensor([[0, 1, 2, 3, 4, 5, 0]])
    >>> dist = torch.randn(1, 6, 6)
    >>> opt_tours = vectorized_type_iii_unstringing(tours, dist)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_type_iii_unstringing(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    max_iterations: int = 50,
    sample_size: int = 50,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized Type III Unstringing local search across a batch of tours.

    Identifies nodes to remove and reinsert in a 4-edge reconnection pattern
    that reverses segments to minimize total distance.

    Args:
        tours: Batch of node sequences of shape [B, N].
        distance_matrix: Pairwise node distances of shape [B, N, N] or [N, N].
        max_iterations: Maximum number of improvement passes.
        sample_size: Number of (k, j, l) tuples to evaluate per node.
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
    if N < 8:
        return tours if is_batch else tours.squeeze(0)

    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        improved_any = False
        for b in range(B):
            tour = tours[b]
            dist = distance_matrix[b]

            valid_indices = torch.where((tour > 0) & (tour < N))[0]
            if len(valid_indices) < 6:
                continue

            best_delta, best_move = _find_best_type_iii_move(tour, dist, valid_indices, sample_size, device, generator)

            if best_move is not None:
                i, k, j, l = best_move
                tours[b] = _apply_type_iii_move(tour, i, k, j, l)
                improved_any = True

        if not improved_any:
            break

    return tours if is_batch else tours.squeeze(0)


def _find_best_type_iii_move(
    tour: torch.Tensor,
    dist: torch.Tensor,
    valid_indices: torch.Tensor,
    sample_size: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    """Finds the best Type III unstringing move for a single tour.

    Args:
        tour: Sequence of shape [N].
        dist: Cost matrix of shape [N, N].
        valid_indices: Nodes available for move of shape [V].
        sample_size: Search depth (number of randomized tuples).
        device: Execution device hardware placement.
        generator: RNG provider.

    Returns:
        Tuple[float, Optional[Tuple[int, int, int, int]]]: (best delta, move params).
    """
    N = len(tour)
    best_delta = 0.0
    best_move = None
    n_valid = len(valid_indices)

    for i_idx in range(n_valid):
        i = int(valid_indices[i_idx].item())

        # Sample or enumerate (k, j, l)
        if sample_size > 0:
            n_samples = min(sample_size, n_valid**3)
            k_s = torch.randint(0, n_valid, (n_samples,), device=device, generator=generator)
            j_s = torch.randint(0, n_valid, (n_samples,), device=device, generator=generator)
            l_s = torch.randint(0, n_valid, (n_samples,), device=device, generator=generator)
            # Valid Type III: usually i < k < j < l in circular order
            valid = (k_s > i_idx) & (j_s > k_s) & (l_s > j_s)
            k_s, j_s, l_s = k_s[valid], j_s[valid], l_s[valid]
        else:
            # Exhaustive would be O(N^4) - skip for now or use original logic
            continue

        for ks, js, ls in zip(k_s, j_s, l_s, strict=False):
            k, j, l = (
                int(valid_indices[ks].item()),
                int(valid_indices[js].item()),
                int(valid_indices[ls].item()),
            )
            delta = _evaluate_type_iii_move(tour, dist, i, k, j, l, N)
            if delta < best_delta - IMPROVEMENT_EPSILON:
                best_delta, best_move = delta, (i, k, j, l)

    return best_delta, best_move


def _evaluate_type_iii_move(tour: torch.Tensor, dist: torch.Tensor, i: int, k: int, j: int, l: int, N: int) -> float:
    """Calculates the delta cost for a specific Type III move.

    Args:
        tour: Node sequence identifiers.
        dist: Pairwise distance weights.
        i: Target position for the unstrung node.
        k: First break point segment split.
        j: Second break point segment split.
        l: Third break point segment split.
        N: Tour length metadata.

    Returns:
        float: Total tour cost change.
    """
    v_ip, v_i, v_in = (
        int(tour[i - 1 if i > 0 else N - 1].item()),
        int(tour[i].item()),
        int(tour[(i + 1) % N].item()),
    )
    v_k, v_kn = int(tour[k].item()), int(tour[(k + 1) % N].item())
    v_j, v_jn = int(tour[j].item()), int(tour[(j + 1) % N].item())
    v_l, v_ln = int(tour[l].item()), int(tour[(l + 1) % N].item())

    # Pattern III: deleted 5 edges, inserted 4 back in (removing node Vi)
    removed = dist[v_ip, v_i] + dist[v_i, v_in] + dist[v_k, v_kn] + dist[v_j, v_jn] + dist[v_l, v_ln]
    inserted = dist[v_ip, v_k] + dist[v_in, v_j] + dist[v_kn, v_l] + dist[v_jn, v_ln]
    return float((inserted - removed).item())


def _apply_type_iii_move(tour: torch.Tensor, i: int, k: int, j: int, l: int) -> torch.Tensor:
    """Applies a Type III unstringing move to the tour.

    Args:
        tour: Original node sequence.
        i: Position of the node to move.
        k: First break coordinate.
        j: Second break coordinate.
        l: Third break coordinate.

    Returns:
        torch.Tensor: Modified tour after segment reversals and re-insertion.
    """
    n = len(tour)
    tl = tour.tolist()

    i_prev = (i - 1) if i > 0 else (n - 1)
    i_next, k_next, j_next, l_next = (i + 1) % n, (k + 1) % n, (j + 1) % n, (l + 1) % n

    # Segments
    s1 = tl[i_next : k + 1] if i_next <= k else tl[i_next:] + tl[: k + 1]
    s2 = tl[k_next : j + 1] if k_next <= j else tl[k_next:] + tl[: j + 1]
    s3 = tl[j_next : l + 1] if j_next <= l else tl[j_next:] + tl[: l + 1]
    rem = tl[l_next : i_prev + 1] if l_next <= i_prev else tl[l_next:] + tl[: i_prev + 1]

    # Pattern: v_{i-1} -> s1_rev -> s2_rev -> s3_rev -> rem
    new_tour = [tl[i_prev]] + s1[::-1] + s2[::-1] + s3[::-1] + rem
    if 0 in new_tour:
        d_idx = new_tour.index(0)
        new_tour = new_tour[d_idx:] + new_tour[:d_idx]

    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
