"""Vectorized linear split algorithm.

This module provides an efficient, GPU-accelerated implementation of the Split
algorithm, which partitions a giant tour into a set of feasible routes using
shortest path logic on a directed acyclic graph (DAG).

Attributes:
    vectorized_linear_split: Splits giant tours into feasible routes.

Example:
    >>> from logic.src.policies.vector.shared.linear import vectorized_linear_split
    >>> routes, costs = vectorized_linear_split(giant_tours, dist_matrix, wastes, vehicle_capacity)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, cast

import torch

from .limited import vectorized_split_limited
from .reconstruction import reconstruct_routes


def vectorized_linear_split(
    giant_tours: torch.Tensor,
    dist_matrix: torch.Tensor,
    wastes: torch.Tensor,
    vehicle_capacity: Union[float, torch.Tensor],
    max_len: Optional[int] = None,
    max_vehicles: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partitions giant tours into feasible routes using shortest paths.

    Computes the optimal segmentation of node sequences into routes subject to
    capacity constraints. If `max_vehicles` is provided, it falls back to the
    limited split variant.

    Args:
        giant_tours: Node sequences of shape [B, N].
        dist_matrix: Pairwise distances of shape [B, N_all, N_all] or [N_all, N_all].
        wastes: Node waste amounts of shape [B, N_all] or [N_all].
        vehicle_capacity: Capacity limit (scalar or [B] tensor).
        max_len: Maximum route length to evaluate for speed.
        max_vehicles: Optional limit on the number of allowed routes.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - routes: Padded route sequences of shape [B, K, N].
            - costs: Total split cost per batch instance of shape [B].
    """
    B, N = giant_tours.size()
    device = giant_tours.device

    if max_len is None:
        max_len = N

    # Standardize inputs
    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    if wastes.dim() == 1:
        wastes = wastes.unsqueeze(0).expand(B, -1)

    if isinstance(vehicle_capacity, torch.Tensor) and vehicle_capacity.dim() == 1:
        vehicle_capacity = vehicle_capacity.unsqueeze(1)

    # Precompute accumulations
    tour_wastes = torch.gather(wastes, 1, giant_tours)
    cum_load = torch.cumsum(tour_wastes, dim=1)

    from_nodes = giant_tours[:, :-1]
    to_nodes = giant_tours[:, 1:]
    batch_ids = torch.arange(B, device=device).view(B, 1)
    tour_dists = dist_matrix[batch_ids, from_nodes, to_nodes]
    tour_dists = torch.cat([torch.zeros((B, 1), device=device), tour_dists], dim=1)
    cum_dist = torch.cumsum(tour_dists, dim=1)

    d_0_i = dist_matrix[batch_ids, 0, giant_tours]
    d_i_0 = dist_matrix[batch_ids, giant_tours, 0]

    cum_load_pad = torch.cat([torch.zeros((B, 1), device=device), cum_load], dim=1)
    cum_dist_pad = torch.cat([torch.zeros((B, 1), device=device), cum_dist], dim=1)

    # Check for max_vehicles constraint
    if max_vehicles and max_vehicles > 0:
        return vectorized_split_limited(
            B,
            N,
            device,
            max_vehicles,
            cast(float, vehicle_capacity),
            cum_load_pad,
            cum_dist_pad,
            d_0_i,
            d_i_0,
            giant_tours,
        )

    # 2. Bellman-Ford / Shortest Path on DAG (Unlimited Vehicles)
    V = torch.full((B, N + 1), float("inf"), device=device)
    V[:, 0] = 0
    P = torch.full((B, N + 1), -1, dtype=torch.long, device=device)

    for i in range(1, N + 1):
        j_start = max(0, i - max_len)
        j_end = i

        js = torch.arange(j_start, j_end, device=device).view(1, -1).expand(B, -1)

        loads = cum_load_pad[:, i : i + 1] - torch.gather(cum_load_pad, 1, js)
        mask = loads <= vehicle_capacity

        dist_0_first = torch.gather(d_0_i, 1, js)
        dist_last_0 = d_i_0[:, i - 1 : i]
        path_dists = cum_dist_pad[:, i : i + 1] - torch.gather(cum_dist_pad, 1, js + 1)
        segment_costs = dist_0_first + path_dists + dist_last_0

        total_costs = torch.gather(V, 1, js) + segment_costs
        total_costs[~mask] = float("inf")

        min_vals, min_idxs = torch.min(total_costs, dim=1)
        V[:, i] = min_vals

        best_js = torch.gather(js, 1, min_idxs.unsqueeze(1)).squeeze(1)
        P[:, i] = best_js

    # 3. Path Reconstruction
    return reconstruct_routes(B, N, giant_tours, P, V[:, N])
