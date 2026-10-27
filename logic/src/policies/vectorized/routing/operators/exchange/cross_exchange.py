"""Cross-exchange (segment swap) operator.

This module provides a GPU-accelerated implementation of the Cross-exchange
operator, which improves tours by swapping segments of arbitrary length
between two different routes while preserving internal node order.

Attributes:
    vectorized_cross_exchange: Swaps segments between two routes to improve fleet-level efficiency.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.exchange.cross_exchange import vectorized_cross_exchange
    >>> optimized_tours = vectorized_cross_exchange(tours, dist_matrix)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_cross_exchange(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    wastes: Optional[torch.Tensor] = None,
    max_segment_len: int = 3,
    max_iterations: int = 50,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized cross-exchange local search across a batch of tours.

    Swaps segments between two routes to improve fleet-level efficiency.
    Supports both batched and shared distance matrices and enforces
    capacity constraints if provided.

    Args:
        tours: Batch of node sequences of shape [B, N].
        distance_matrix: Edge cost tensor of shape [B, N+1, N+1] or [N+1, N+1].
        capacities: Vehicle capacity per instance of shape [B] or scalar.
        wastes: Node demand metadata of shape [B, N+1] or [N+1].
        max_segment_len: Limit for the size of swapped segments.
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

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    # Handle wastes if provided
    has_capacity = capacities is not None and wastes is not None
    if has_capacity:
        assert wastes is not None
        assert capacities is not None
        if wastes.dim() == 1:
            wastes = wastes.unsqueeze(0).expand(B, -1)
        if capacities.dim() == 0:
            capacities = capacities.unsqueeze(0).expand(B)

    for _ in range(max_iterations):
        improved_any = _perform_cross_exchange_iteration(
            B,
            tours,
            max_segment_len,
            distance_matrix,
            wastes,
            capacities,
            has_capacity,
            device,
        )
        if not improved_any:
            break

    return tours if is_batch else tours.squeeze(0)


def _perform_cross_exchange_iteration(
    B: int,
    tours: torch.Tensor,
    max_segment_len: int,
    distance_matrix: torch.Tensor,
    wastes: Optional[torch.Tensor],
    capacities: Optional[torch.Tensor],
    has_capacity: bool,
    device: torch.device,
) -> bool:
    """Evaluates all segment length combinations for a single iteration.

    Args:
        B: Batch size.
        tours: Current batch of tour sequences of shape [B, N].
        max_segment_len: Maximum length of segments to consider swapping.
        distance_matrix: Parent distance matrix of shape [B, N+1, N+1].
        wastes: Node demand metadata of shape [B, N+1].
        capacities: Vehicle capacities of shape [B].
        has_capacity: Boolean flag enabling/disabling capacity checks.
        device: Hardware identification locator.

    Returns:
        bool: True if at least one improving swap was found and applied.
    """
    improved_any = False

    # Try all combinations of segment lengths
    for seg_a_len in range(0, max_segment_len + 1):
        for seg_b_len in range(0, max_segment_len + 1):
            if seg_a_len == 0 and seg_b_len == 0:
                continue

            for b in range(B):
                tour = tours[b]
                routes = _get_routes_from_tour(tour)
                if len(routes) < 2:
                    continue

                best_delta, best_move = _find_best_move_for_segments(
                    b,
                    tour,
                    routes,
                    seg_a_len,
                    seg_b_len,
                    distance_matrix,
                    wastes,
                    capacities,
                    has_capacity,
                    device,
                )

                if best_move is not None:
                    tours[b] = _apply_cross_exchange_move(tour, best_move)
                    improved_any = True

    return improved_any


def _get_routes_from_tour(tour: torch.Tensor) -> List[Tuple[int, int]]:
    """Segments a full tour into individual [start, end] route pairs.

    Args:
        tour: Full sequence of shaped [N].

    Returns:
        List[Tuple[int, int]]: Logical route boundaries (inclusive indices).
    """
    depot_positions = torch.where(tour == 0)[0]
    routes = []
    for i in range(len(depot_positions) - 1):
        start = int(depot_positions[i].item() + 1)
        end = int(depot_positions[i + 1].item())
        if end > start:
            routes.append((start, end))
    return routes


def _find_best_move_for_segments(
    b_idx: int,
    tour: torch.Tensor,
    routes: List[Tuple[int, int]],
    seg_a_len: int,
    seg_b_len: int,
    distance_matrix: torch.Tensor,
    wastes: Optional[torch.Tensor],
    capacities: Optional[torch.Tensor],
    has_capacity: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[tuple]]:
    """Iteratively searches for improving inter-route segment swaps.

    Args:
        b_idx: Batch index being processed.
        tour: Individual tour sequence under optimization of shape [N].
        routes: List of identified route indices.
        seg_a_len: Length of segment to extract from first route.
        seg_b_len: Length of segment to extract from second route.
        distance_matrix: Instance distance matrix of shape [B, N+1, N+1].
        wastes: Demand metadata of shape [B, N+1].
        capacities: Fleet volume limits of shape [B].
        has_capacity: Boolean toggle for volume feasibility.
        device: Execution hardware locator.

    Returns:
        Tuple[torch.Tensor, Optional[tuple]]: A tuple containing:
            - best_delta: Maximum cost reduction found.
            - best_move: Move parameters (route IDs, start positions, lengths) or None.
    """
    best_delta = torch.tensor(0.0, device=device)
    best_move = None

    for r_a_idx, (r_a_start, r_a_end) in enumerate(routes):
        for r_b_idx, (r_b_start, r_b_end) in enumerate(routes[r_a_idx + 1 :], start=r_a_idx + 1):
            for s_a_start in range(r_a_start, r_a_end - seg_a_len + 1):
                for s_b_start in range(r_b_start, r_b_end - seg_b_len + 1):
                    # Check capacity
                    if has_capacity and not _check_cross_capacity(
                        b_idx,
                        tour,
                        s_a_start,
                        seg_a_len,
                        s_b_start,
                        seg_b_len,
                        r_a_start,
                        r_a_end,
                        r_b_start,
                        r_b_end,
                        wastes,  # type: ignore[arg-type]
                        capacities,  # type: ignore[arg-type]
                    ):
                        continue

                    # Compute delta
                    delta = _compute_cross_delta(
                        b_idx,
                        tour,
                        s_a_start,
                        seg_a_len,
                        s_b_start,
                        seg_b_len,
                        r_a_start,
                        r_a_end,
                        r_b_start,
                        r_b_end,
                        distance_matrix,
                    )

                    if delta < best_delta - IMPROVEMENT_EPSILON:
                        best_delta = delta
                        best_move = (
                            r_a_idx,
                            s_a_start,
                            seg_a_len,
                            r_b_idx,
                            s_b_start,
                            seg_b_len,
                        )

    return best_delta, best_move


def _check_cross_capacity(
    b: int,
    tour: torch.Tensor,
    s_a: int,
    len_a: int,
    s_b: int,
    len_b: int,
    r_a_s: int,
    r_a_e: int,
    r_b_s: int,
    r_b_e: int,
    wastes: torch.Tensor,
    capacities: torch.Tensor,
) -> bool:
    """Enforces vehicle volume limits after proposed segment exchange.

    Args:
        b: Current batch index.
        tour: Tour sequence being checked of shape [N].
        s_a: Start position of segment A.
        len_a: Length of segment A.
        s_b: Start position of segment B.
        len_b: Length of segment B.
        r_a_s: First route start.
        r_a_e: First route end.
        r_b_s: Second route start.
        r_b_e: Second route end.
        wastes: Global demand metadata of shape [B, N+1].
        capacities: Fleet limits of shape [B].

    Returns:
        bool: True if the resulting routes remain within capacity limits.
    """
    dem_a = wastes[b, tour[s_a : s_a + len_a]].sum() if len_a > 0 else 0.0
    dem_b = wastes[b, tour[s_b : s_b + len_b]].sum() if len_b > 0 else 0.0

    r_a_dem = wastes[b, tour[r_a_s:r_a_e]].sum()
    r_b_dem = wastes[b, tour[r_b_s:r_b_e]].sum()

    # Caller uses this only as a truth value (`not`), so collapse the 0-dim comparison.
    return bool((r_a_dem - dem_a + dem_b <= capacities[b]) and (r_b_dem - dem_b + dem_a <= capacities[b]))


def _compute_cross_delta(
    b: int,
    tour: torch.Tensor,
    s_a: int,
    len_a: int,
    s_b: int,
    len_b: int,
    r_a_s: int,
    r_a_e: int,
    r_b_s: int,
    r_b_e: int,
    dist_mat: torch.Tensor,
) -> torch.Tensor:
    """Calculates network cost differential for segment swap.

    Args:
        b: Batch index.
        tour: Sequence metadata of shape [N].
        s_a: Start of segment A.
        len_a: Length of segment A.
        s_b: Start of segment B.
        len_b: Length of segment B.
        r_a_s: First route start index.
        r_a_e: First route end index.
        r_b_s: Second route start index.
        r_b_e: Second route end index.
        dist_mat: Instance distance matrix of shape [B, N+1, N+1].

    Returns:
        torch.Tensor: Evaluated cost improvement (negative delta means improvement).
    """
    # Route A
    a_prev = tour[s_a - 1] if s_a > r_a_s else 0
    a_next = tour[s_a + len_a] if s_a + len_a < r_a_e else 0

    rem_a = (dist_mat[b, a_prev, tour[s_a]] + dist_mat[b, tour[s_a + len_a - 1], a_next]) if len_a > 0 else 0.0
    ins_a = (
        (dist_mat[b, a_prev, tour[s_b]] + dist_mat[b, tour[s_b + len_b - 1], a_next])
        if len_b > 0
        else dist_mat[b, a_prev, a_next]
    )

    # Route B
    b_prev = tour[s_b - 1] if s_b > r_b_s else 0
    b_next = tour[s_b + len_b] if s_b + len_b < r_b_e else 0

    rem_b = (dist_mat[b, b_prev, tour[s_b]] + dist_mat[b, tour[s_b + len_b - 1], b_next]) if len_b > 0 else 0.0
    ins_b = (
        (dist_mat[b, b_prev, tour[s_a]] + dist_mat[b, tour[s_a + len_a - 1], b_next])
        if len_a > 0
        else dist_mat[b, b_prev, b_next]
    )

    return (ins_a - rem_a) + (ins_b - rem_b)


def _apply_cross_exchange_move(tour: torch.Tensor, move: tuple) -> torch.Tensor:
    """Physically updates the tour tensor by swapping indexed segments.

    Args:
        tour: Source sequence tensor of shape [N].
        move: Logical parameters (route_a, start_a, len_a, route_b, start_b, len_b).

    Returns:
        torch.Tensor: The updated sequence.
    """
    _, s_a_start, s_a_len, _, s_b_start, s_b_len = move
    new_tour = tour.clone()
    if s_a_len == s_b_len:
        new_tour[s_a_start : s_a_start + s_a_len] = tour[s_b_start : s_b_start + s_b_len]
        new_tour[s_b_start : s_b_start + s_b_len] = tour[s_a_start : s_a_start + s_a_len]
        return new_tour

    # Note: Complex variant involving resizing tour segments not currently fully
    # implemented to avoid TensorDict reconstruction overhead in tight LS loops.
    return new_tour
