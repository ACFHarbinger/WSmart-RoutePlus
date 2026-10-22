"""Shaw removal operator.

This module provides a GPU-accelerated implementation of the Shaw removal
heuristic (relatedness-based), which ejects nodes from the tour that are
similar based on distance, demand, and time windows.

Attributes:
    vectorized_shaw_removal: Removes a set of related nodes based on distance,
    time windows, and waste metrics.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.destroy.shaw_removal import vectorized_shaw_removal
    >>> tours, removed_nodes = vectorized_shaw_removal(
    ...     tours, dist_matrix, n_remove, wastes, time_windows
    ... )
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def vectorized_shaw_removal(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    n_remove: int,
    wastes: Optional[torch.Tensor] = None,
    time_windows: Optional[torch.Tensor] = None,
    phi: float = 9.0,
    chi: float = 3.0,
    psi: float = 2.0,
    randomization_factor: float = 2.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized Shaw removal across a batch of tours using PyTorch.

    Shaw removal computes relatedness between nodes based on distance,
    time windows, and waste. Nodes with high similarity to the removed
    set are prioritized for subsequent removal to create clean clusters.

    Args:
        tours: Batch of sequences of shape [B, N].
        distance_matrix: Edge weights of shape [B, N+1, N+1] or [N+1, N+1].
        n_remove: Number of nodes to remove per tour.
        wastes: Node demands of shape [B, N+1] or [N+1].
        time_windows: Windows metadata of shape [B, N+1, 2] or [N+1, 2].
        phi: Weight for distance relatedness.
        chi: Weight for time window relatedness.
        psi: Weight for waste relatedness.
        randomization_factor: Power for randomized selection (p-value).
        generator: Torch device-side RNG.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - modified_tours: Batch of tours with removed nodes marked (shape [B, N]).
            - removed_nodes: IDs of the removed nodes per instance (shape [B, n_remove]).
    """
    # Prepare inputs
    tours, distance_matrix, wastes, time_windows, is_batch = _prepare_shaw_inputs(
        tours, distance_matrix, wastes, time_windows
    )
    device = tours.device

    # Initialize parameters and tracks
    (
        distance_matrix,
        wastes,
        time_windows,
        max_dist,
        max_waste,
        max_time,
    ) = _init_shaw_params(tours, distance_matrix, wastes, time_windows)
    B, N = tours.shape
    removed_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    removed_list = torch.full((B, n_remove), -1, dtype=torch.long, device=device)
    removed_count = torch.zeros(B, dtype=torch.long, device=device)

    valid_mask = tours >= 0
    valid_counts = valid_mask.sum(dim=1)

    # Random seed selection
    _select_seed_nodes(
        B,
        valid_mask,
        valid_counts,
        removed_mask,
        removed_list,
        removed_count,
        device,
        generator,
    )

    # Iteratively select related nodes
    for _ in range(1, n_remove):
        relatedness_sum = _calculate_relatedness_batch(
            B,
            N,
            tours,
            distance_matrix,
            wastes,
            time_windows,
            removed_mask,
            removed_count,
            valid_counts,
            max_dist,
            max_waste,
            max_time,
            phi,
            psi,
            chi,
            device,
        )

        relatedness_scores = relatedness_sum
        relatedness_scores[removed_mask] = float("inf")
        relatedness_scores[~valid_mask] = float("inf")

        _select_next_removal_batch(
            B,
            n_remove,
            randomization_factor,
            relatedness_scores,
            removed_mask,
            removed_list,
            removed_count,
            valid_mask,
            valid_counts,
            device,
            generator,
        )

    # Create modified tours with removed nodes marked as -1
    modified_tours = tours.clone()
    for b in range(B):
        modified_tours[b, removed_mask[b]] = -1

    return (
        modified_tours if is_batch else modified_tours.squeeze(0),
        removed_list if is_batch else removed_list.squeeze(0),
    )


def _prepare_shaw_inputs(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    wastes: Optional[torch.Tensor],
    time_windows: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    bool,
]:
    """Ensures all inputs are batched tensors.

    Args:
        tours: Input tour sequences.
        distance_matrix: Global distance metadata.
        wastes: Optional demand metadata.
        time_windows: Optional temporal metadata.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], bool]:
            Batched tours, distance_matrix, wastes, time_windows, and a batch flag.
    """
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)
        if distance_matrix.dim() == 3:
            distance_matrix = distance_matrix.unsqueeze(0)
        if wastes is not None and wastes.dim() == 2:
            wastes = wastes.unsqueeze(0)
        if time_windows is not None and time_windows.dim() == 3:
            time_windows = time_windows.unsqueeze(0)
    return tours, distance_matrix, wastes, time_windows, is_batch


def _select_seed_nodes(
    B: int,
    valid_mask: torch.Tensor,
    valid_counts: torch.Tensor,
    removed_mask: torch.Tensor,
    removed_list: torch.Tensor,
    removed_count: torch.Tensor,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> None:
    """Selects a random seed node for each tour.

    Args:
        B: Batch instances count.
        valid_mask: Boolean mask of non-depot nodes.
        valid_counts: Total valid nodes per tour.
        removed_mask: Target boolean tracker for removals.
        removed_list: Ordered list of removed node IDs.
        removed_count: Running total of removals per instance.
        device: Hardware identification locator.
        generator: PyTorch random number generator.
    """
    for b in range(B):
        if valid_counts[b] > 0:
            valid_indices = torch.where(valid_mask[b])[0]
            seed_idx = int(
                valid_indices[torch.randint(len(valid_indices), (1,), device=device, generator=generator)].item()
            )
            removed_mask[b, seed_idx] = True
            removed_list[b, 0] = seed_idx
            removed_count[b] = 1


def _calculate_relatedness_batch(
    B: int,
    N: int,
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    wastes: Optional[torch.Tensor],
    time_windows: Optional[torch.Tensor],
    removed_mask: torch.Tensor,
    removed_count: torch.Tensor,
    valid_counts: torch.Tensor,
    max_dist: float,
    max_waste: float,
    max_time: float,
    phi: float,
    psi: float,
    chi: float,
    device: torch.device,
) -> torch.Tensor:
    """Calculates relatedness scores for all nodes against removed nodes.

    Args:
        B: Batch instances count.
        N: Full sequence length.
        tours: Current sequences of shape [B, N].
        distance_matrix: Distance metadata of shape [B, N+1, N+1].
        wastes: Optional demand metadata.
        time_windows: Optional temporal metadata.
        removed_mask: Tracker for already removed nodes.
        removed_count: Count of nodes already removed per instance.
        valid_counts: Number of non-depot nodes per instance.
        max_dist: Maximum pairwise distance (for normalization).
        max_waste: Maximum node demand.
        max_time: Maximum earliest temporal window.
        phi: Distance similarity weight.
        psi: Waste similarity weight.
        chi: Temporal similarity weight.
        device: Hardware identification locator.

    Returns:
        torch.Tensor: Computed relatedness score tensor of shape [B, N].
    """
    relatedness_sum = torch.zeros((B, N), device=device)

    for b in range(B):
        if removed_count[b] >= valid_counts[b]:
            continue

        rem_nodes = tours[b, removed_mask[b]]  # [n_rem]

        # Distance component
        d_rel = distance_matrix[b][tours[b].unsqueeze(1), rem_nodes.unsqueeze(0)] / max_dist
        rel = phi * d_rel

        # Waste component
        if wastes is not None:
            q_rel = torch.abs(wastes[b][tours[b]].unsqueeze(1) - wastes[b][rem_nodes].unsqueeze(0)) / max_waste
            rel += psi * q_rel

        # Time window component
        if time_windows is not None:
            t_rel = (
                torch.abs(time_windows[b][tours[b], 0].unsqueeze(1) - time_windows[b][rem_nodes, 0].unsqueeze(0))
                / max_time
            )
            rel += chi * t_rel

        relatedness_sum[b] = rel.mean(dim=1)

    return relatedness_sum


def _select_next_removal_batch(
    B: int,
    n_remove: int,
    randomization_factor: float,
    relatedness_scores: torch.Tensor,
    removed_mask: torch.Tensor,
    removed_list: torch.Tensor,
    removed_count: torch.Tensor,
    valid_mask: torch.Tensor,
    valid_counts: torch.Tensor,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> None:
    """Selects the next node to remove using randomized power law.

    Args:
        B: Batch instances count.
        n_remove: Target number of removals.
        randomization_factor: Selection stochasticity (p-value).
        relatedness_scores: Precomputed similarity tensor of shape [B, N].
        removed_mask: Boolean tracker for exclusions.
        removed_list: Ordered list of removals IDs.
        removed_count: Running total of removals.
        valid_mask: Boolean identifier for non-depot nodes.
        valid_counts: Total count of valid nodes per sequence.
        device: Hardware identification locator.
        generator: PyTorch random number generator.
    """
    for b in range(B):
        if removed_count[b] >= n_remove or removed_count[b] >= valid_counts[b]:
            continue

        # Get candidates (non-removed, non-padding, finite relatedness)
        candidates_mask = (~removed_mask[b]) & valid_mask[b] & (relatedness_scores[b] < float("inf"))

        if not candidates_mask.any():
            break

        candidate_indices = torch.where(candidates_mask)[0]
        candidate_scores = relatedness_scores[b, candidates_mask]

        # Sort by relatedness (lower = more related)
        sorted_indices = torch.argsort(candidate_scores)
        sorted_candidates = candidate_indices[sorted_indices]

        # Randomized selection using power law
        y = torch.rand(1, device=device, generator=generator).item()
        idx = int((y**randomization_factor) * len(sorted_candidates))
        idx = min(idx, len(sorted_candidates) - 1)

        selected_node_idx = sorted_candidates[idx]

        # Mark as removed
        removed_mask[b, selected_node_idx] = True
        removed_list[b, int(removed_count[b].item())] = selected_node_idx
        removed_count[b] += 1


def _init_shaw_params(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    wastes: Optional[torch.Tensor],
    time_windows: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, float, float]:
    """Initialize max values for normalization.

    Args:
        tours: Current sequences of shape [B, N].
        distance_matrix: Global distance weights.
        wastes: Optional demand metadata.
        time_windows: Optional temporal metadata.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], float, float, float]:
            Processed distance_matrix, wastes, time_windows, max_dist, max_waste, and max_time.
    """
    B, _ = tours.shape

    # Ensure distance matrix is batched
    if distance_matrix.dim() == 2:
        distance_matrix = distance_matrix.unsqueeze(0).expand(B, -1, -1)

    max_dist = float(distance_matrix.max().item())
    max_dist = max_dist if max_dist > 1e-6 else 1.0

    max_waste = 1.0
    if wastes is not None:
        if wastes.dim() == 1:
            wastes = wastes.unsqueeze(0).expand(B, -1)
        max_waste = float(wastes.max().item())
        max_waste = max_waste if max_waste > 1e-6 else 1.0

    max_time = 1.0
    if time_windows is not None:
        if time_windows.dim() == 2:
            time_windows = time_windows.unsqueeze(0).expand(B, -1, -1)
        # Max of earliest times
        max_time = float(time_windows[:, :, 0].max().item())
        max_time = max_time if max_time > 1e-6 else 1.0

    return distance_matrix, wastes, time_windows, max_dist, max_waste, max_time
