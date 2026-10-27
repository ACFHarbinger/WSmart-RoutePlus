"""Or-opt local search operator.

This module provides a GPU-accelerated implementation of the Or-opt operator,
which improves tours by relocating clusters of consecutive nodes (typically 1-3)
to more efficient positions across routes.

Attributes:
    vectorized_or_opt: Systematically evaluates the relocation of node
    sequences of length 'k' to every possible insertion point in the tour,
    utilizing GPU parallelism to find the global best move per tour per iteration.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.exchange.or_opt import vectorized_or_opt
    >>> optimized_tours = vectorized_or_opt(tours, dist_matrix, capacities, wastes, chain_lengths, max_iterations)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_or_opt(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    wastes: Optional[torch.Tensor] = None,
    chain_lengths: Tuple[int, ...] = (1, 2, 3),
    max_iterations: int = 100,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized Or-opt local search using parallel chain evaluation.

    Systematically evaluates the relocation of node sequences of length 'k'
    to every possible insertion point in the tour, utilizing GPU parallelism
    to find the global best move per tour per iteration.

    Args:
        tours: Batch of node sequences of shape [B, N].
        distance_matrix: Edge cost tensor of shape [B, N+1, N+1] or [N+1, N+1].
        capacities: Vehicle limits per instance of shape [B] or scalar.
        wastes: Node demand metadata of shape [B, N+1] or [N+1].
        chain_lengths: Collection of cluster sizes to attempt.
        max_iterations: Iteration limit for monotonic improvement.
        generator: Torch device-side RNG (for future randomness).

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

    batch_indices = torch.arange(B, device=device).view(B, 1)

    # Iterate through chain lengths
    for chain_len in chain_lengths:
        if chain_len >= N // 2:
            continue

        for _ in range(max_iterations):
            improved_any = False
            chain_starts = torch.arange(N - chain_len, device=device)
            n_chains = len(chain_starts)
            if n_chains == 0:
                break

            chain_starts_batch = chain_starts.unsqueeze(0).expand(B, -1)

            # Compute removal gain and chain wastes
            removal_gain, chain_wastes = _compute_removal_info(
                tours,
                chain_starts_batch,
                chain_len,
                distance_matrix,
                wastes,
                has_capacity,
                batch_indices,
                B,
                n_chains,
                N,
            )

            # Find best insertions
            best_delta, best_insert_pos = _find_best_insertions(
                B,
                n_chains,
                N,
                chain_len,
                chain_starts,
                tours,
                distance_matrix,
                removal_gain,
                chain_wastes,
                capacities,
                has_capacity,
                batch_indices,
                device,
            )

            # Find best chain to move for each batch instance
            best_chain_delta, best_chain_idx = torch.min(best_delta, dim=1)

            improved = best_chain_delta < -IMPROVEMENT_EPSILON
            if not improved.any():
                break

            improved_any = True
            tours = _apply_or_opt_moves(
                tours,
                improved,
                best_chain_idx,
                best_insert_pos,
                chain_starts,
                chain_len,
                B,
                N,
                device,
            )

            if not improved_any:
                break

    return tours if is_batch else tours.squeeze(0)


def _compute_removal_info(
    tours: torch.Tensor,
    chain_starts: torch.Tensor,
    chain_len: int,
    dist_mat: torch.Tensor,
    wastes: Optional[torch.Tensor],
    has_capacity: bool,
    batch_indices: torch.Tensor,
    B: int,
    n_chains: int,
    N: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Calculates removal gain and sequence demand weights.

    Args:
        tours: Batch of node sequences of shape [B, N].
        chain_starts: Matrix of prospective cluster start positions of shape [B, K].
        chain_len: Number of nodes in the moving cluster.
        dist_mat: Global distance matrix of shape [B, N+1, N+1].
        wastes: Demand metadata of shape [B, N+1] or None.
        has_capacity: Boolean threshold flag.
        batch_indices: Unit batch index column vector of shape [B, 1].
        B: Determined batch size.
        n_chains: Count of evaluation chains.
        N: Full sequence length.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
            - removal_gain: Cost differential from removing the cluster.
            - chain_wastes: Aggregate demand for each identified moving cluster.
    """
    prev_idx = (chain_starts - 1).clamp(min=0)
    next_idx = (chain_starts + chain_len).clamp(max=N - 1)

    prev_nodes = torch.gather(tours, 1, prev_idx)
    chain_first = torch.gather(tours, 1, chain_starts)
    chain_last = torch.gather(tours, 1, (chain_starts + chain_len - 1).clamp(max=N - 1))
    next_nodes = torch.gather(tours, 1, next_idx)

    b_exp = batch_indices.expand(B, n_chains)
    removal_gain = (
        dist_mat[b_exp, prev_nodes, chain_first]
        + dist_mat[b_exp, chain_last, next_nodes]
        - dist_mat[b_exp, prev_nodes, next_nodes]
    )

    chain_wastes = None
    if has_capacity:
        chain_wastes = torch.zeros(B, n_chains, device=tours.device)
        for k in range(chain_len):
            nodes = torch.gather(tours, 1, (chain_starts + k).clamp(max=N - 1))
            chain_wastes += torch.gather(wastes, 1, nodes)  # type: ignore[arg-type]

    return removal_gain, chain_wastes


def _find_best_insertions(
    B: int,
    n_chains: int,
    N: int,
    chain_len: int,
    chain_starts: torch.Tensor,
    tours: torch.Tensor,
    dist_mat: torch.Tensor,
    rem_gain: torch.Tensor,
    chain_wastes: Optional[torch.Tensor],
    caps: Optional[torch.Tensor],
    has_cap: bool,
    b_idx: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exhaustively searches for the best insertion point per chain.

    Args:
        B: Batch instances count.
        n_chains: Number of parallel clusters being evaluated.
        N: Full sequence length.
        chain_len: Relocation cluster size.
        chain_starts: Coordinate indices for each prospective cluster.
        tours: Global node sequences of shape [B, N].
        dist_mat: Distance weights of shape [B, N+1, N+1].
        rem_gain: Calculated savings from removing clusters.
        chain_wastes: Precalculated cluster demands.
        caps: Fleet capacity limits of shape [B].
        has_cap: Boolean feasibility flag.
        b_idx: Batch index column vector.
        device: Hardware identification locator.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - best_delta: Minimum cost differential found for each prospective move.
            - best_insert_pos: Optimized insertion coordinate per cluster.
    """
    best_delta = torch.full((B, n_chains), float("inf"), device=device)
    best_insert_pos = torch.zeros((B, n_chains), dtype=torch.long, device=device)
    b_exp = b_idx.expand(B, n_chains)

    chain_first = torch.gather(tours, 1, chain_starts.unsqueeze(0).expand(B, -1))
    chain_last = torch.gather(tours, 1, (chain_starts + chain_len - 1).clamp(max=N - 1).unsqueeze(0).expand(B, -1))

    for insert_pos in range(N + 1):
        ins_prev = torch.full((B, n_chains), insert_pos - 1, device=device).clamp(min=0, max=N - 1)
        ins_next = torch.full((B, n_chains), insert_pos, device=device).clamp(min=0, max=N - 1)

        ins_prev_nodes = torch.gather(tours, 1, ins_prev)
        ins_next_nodes = torch.gather(tours, 1, ins_next)

        skip_mask = (chain_starts.unsqueeze(0) <= insert_pos) & (insert_pos <= chain_starts.unsqueeze(0) + chain_len)

        ins_cost = (
            dist_mat[b_exp, ins_prev_nodes, chain_first]
            + dist_mat[b_exp, chain_last, ins_next_nodes]
            - dist_mat[b_exp, ins_prev_nodes, ins_next_nodes]
        )

        if has_cap:
            assert caps is not None
            feasible = chain_wastes <= caps.unsqueeze(1)  # type: ignore[operator]
            ins_cost = torch.where(feasible, ins_cost, torch.tensor(float("inf"), device=device))

        delta = torch.where(skip_mask, torch.tensor(float("inf"), device=device), ins_cost - rem_gain)

        better = delta < best_delta
        best_delta = torch.where(better, delta, best_delta)
        best_insert_pos = torch.where(better, ins_next, best_insert_pos)

    return best_delta, best_insert_pos


def _apply_or_opt_moves(
    tours: torch.Tensor,
    improved: torch.Tensor,
    best_chain_idx: torch.Tensor,
    best_insert_pos: torch.Tensor,
    chain_starts: torch.Tensor,
    chain_len: int,
    B: int,
    N: int,
    device: torch.device,
) -> torch.Tensor:
    """Updates node order by physically cat-ing segments.

    Args:
        tours: Target sequences of shape [B, N].
        improved: Binary boolean activation mask of shape [B].
        best_chain_idx: Index IDs of winning clusters of shape [B].
        best_insert_pos: Coordinate targets for winners of shape [B, K].
        chain_starts: Global pool of starting positions.
        chain_len: Cluster size.
        B: Batch size.
        N: Sequence length.
        device: Hardware identification locator.

    Returns:
        torch.Tensor: Updated sequence batch.
    """
    for b in range(B):
        if improved[b]:
            idx = int(best_chain_idx[b].item())
            start = int(chain_starts[idx].item())
            pos = int(best_insert_pos[b, idx].item())

            chain = tours[b, start : start + chain_len].clone()
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[start : start + chain_len] = False
            remaining = tours[b][mask]

            adj_pos = pos - chain_len if pos > start + chain_len else pos
            new_tour = torch.cat([remaining[:adj_pos], chain, remaining[adj_pos:]])

            if new_tour.size(0) < N:
                new_tour = torch.cat(
                    [
                        new_tour,
                        torch.zeros(N - new_tour.size(0), dtype=tours.dtype, device=device),
                    ]
                )
            tours[b] = new_tour[:N]
    return tours
