"""Lambda-interchange operator.

This module provides a GPU-accelerated implementation of the Lambda-interchange
operator, a generalized version of cross-exchange that systematically explores
segment swaps of lengths ranging from 0 to λ_max.

Attributes:
    vectorized_lambda_interchange: Performs exhaustive local search over all
    segment combinations (0..λ) for fleet-level improvement.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.exchange.lambda_interchange import vectorized_lambda_interchange
    >>> optimized_tours = vectorized_lambda_interchange(tours, dist_matrix, lambda_max=2)
"""

from __future__ import annotations

from typing import Optional

import torch

from .cross_exchange import vectorized_cross_exchange


def vectorized_lambda_interchange(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    wastes: Optional[torch.Tensor] = None,
    lambda_max: int = 2,
    max_iterations: int = 50,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized λ-interchange local search across a batch of tours.

    Systematically explores the cross-exchange neighborhood for all segment
    length combinations (λ_a, λ_b) where 0 <= λ_i <= λ_max. This exhaustive
    sweep captures moves such as:
    - (0, 1): Node relocation
    - (1, 1): Single node swap
    - (2, 2): Two-node segment exchange

    Args:
        tours: Batch of node sequences of shape [B, N].
        distance_matrix: Edge cost tensor of shape [B, N+1, N+1] or [N+1, N+1].
        capacities: Vehicle capacity per instance of shape [B] or scalar.
        wastes: Node demand metadata of shape [B, N+1] or [N+1].
        lambda_max: Maximum segment length to consider (typically 1-3).
        max_iterations: Limit for neighborhood sweeps (iteration count).
        generator: Torch device-side RNG (for future randomness).

    Returns:
        torch.Tensor: Optimized tours of shape [B, N].
    """
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

    # Main loop: try all segment length combinations
    for _ in range(max_iterations):
        improved_this_sweep = False

        # Systematically explore all (λ_a, λ_b) combinations
        for seg_a_len in range(lambda_max + 1):
            for seg_b_len in range(lambda_max + 1):
                # Skip no-op case
                if seg_a_len == 0 and seg_b_len == 0:
                    continue

                # Store initial tours to detect improvement
                tours_before = tours.clone()

                # Apply cross-exchange with this segment length combination
                tours = vectorized_cross_exchange(
                    tours=tours,
                    distance_matrix=distance_matrix,
                    capacities=capacities,
                    wastes=wastes,
                    max_segment_len=max(seg_a_len, seg_b_len),
                    max_iterations=1,  # Single iteration per combination
                )

                # Check if any improvement occurred
                if not torch.equal(tours, tours_before):
                    improved_this_sweep = True
                    # First-improvement strategy: restart search after improvement
                    break

            if improved_this_sweep:
                break  # Restart from λ=0

        # If no improvement in full sweep, we're done
        if not improved_this_sweep:
            break

    return tours if is_batch else tours.squeeze(0)
