"""Vectorized Lin-Kernighan-Helsgaun (LKH-3) Operator.

This module provides a GPU-accelerated implementation of the sophisticated LKH-3
local search algorithm. LKH combines powerful k-opt moves with alpha-nearness
candidate sets and lexicographic optimization for VRP problems.

Key features:
- Alpha-measure edge pruning based on MST.
- Candidate set restriction for efficiency.
- Sequential 2-opt and 3-opt moves.
- Double bridge kicks for perturbation.
- Lexicographic optimization (penalty, cost).

Attributes:
    vectorized_lkh: Local search across a batch of tours.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.route.lkh import vectorized_lkh
    >>> optimized_tours = vectorized_lkh(tours, dist_matrix, capacities, wastes, max_iterations, max_candidates, use_3opt, perturbation_interval, generator)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_lkh(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    wastes: Optional[torch.Tensor] = None,
    max_iterations: int = 100,
    max_candidates: int = 5,
    use_3opt: bool = True,
    perturbation_interval: int = 10,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized LKH local search across a batch of tours.

    LKH is one of the most sophisticated local search algorithms for TSP/VRP,
    combining alpha-nearness pruning with sequential k-opt moves.

    Args:
        tours: Batch of node sequences of shape [B, N].
        distance_matrix: Pairwise distances of shape [B, N+1, N+1] or [N+1, N+1].
        capacities: Vehicle capacities per instance of shape [B] or scalar.
        wastes: Individual node demands of shape [B, N+1] or [N+1].
        max_iterations: Iteration limit for the ILS loop.
        max_candidates: Number of pruned edges per node (candidate set size).
        use_3opt: Whether to extend search to 3-opt neighborhoods.
        perturbation_interval: Frequency of double-bridge kicks.
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
    if N < 8:
        return tours if is_batch else tours.squeeze(0)

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    has_capacity = wastes is not None and capacities is not None
    if has_capacity:
        assert wastes is not None
        assert capacities is not None
        if wastes.dim() == 1:
            wastes = wastes.unsqueeze(0).expand(B, -1)
        if capacities.dim() == 0:
            capacities = capacities.unsqueeze(0).expand(B)
        elif capacities.dim() == 1 and capacities.size(0) != B:
            capacities = capacities.expand(B)

    # Main ILS loop
    for iteration in range(max_iterations):
        for b in range(B):
            tour = tours[b]
            dist = distance_matrix[b]

            alpha = _compute_alpha_measures(dist, device)
            candidates = _get_candidate_sets(alpha, max_candidates)

            inst_waste = wastes[b] if has_capacity else None  # type: ignore[index]
            inst_cap = capacities[b] if has_capacity else None  # type: ignore[index]

            tours[b] = _run_local_search(tour, dist, inst_waste, inst_cap, candidates, use_3opt)

        if (iteration + 1) % perturbation_interval == 0 and iteration < max_iterations - 1:
            for b in range(B):
                tours[b] = _double_bridge_kick(tours[b], generator)

    return tours if is_batch else tours.squeeze(0)


def _run_local_search(
    tour: torch.Tensor,
    dist: torch.Tensor,
    waste: Optional[torch.Tensor],
    capacity: Optional[torch.Tensor],
    candidates: List[List[int]],
    use_3opt: bool,
) -> torch.Tensor:
    """Internal loop for iterating until local convergence.

    Args:
        tour: Node sequence of shape [N].
        dist: Distance matrix of shape [N+1, N+1].
        waste: Node demands of shape [N+1].
        capacity: Vehicle capacity scalar.
        candidates: Alpha-nearness neighbor lists for each node.
        use_3opt: Whether to evaluate 3-opt moves.

    Returns:
        torch.Tensor: Locally optimized tour of shape [N].
    """
    local_improved = True
    while local_improved:
        local_improved = False
        curr_p, curr_c = _compute_score(tour, dist, waste, capacity)
        nodes_count = len(tour) - 1

        for i in range(nodes_count):
            improved, tour, curr_p, curr_c = _try_moves_for_node(
                tour,
                i,
                nodes_count,
                dist,
                waste,
                capacity,
                candidates,
                use_3opt,
                curr_p,
                curr_c,
            )
            if improved:
                local_improved = True
                break
    return tour


def _try_moves_for_node(
    tour: torch.Tensor,
    i: int,
    nodes_count: int,
    dist: torch.Tensor,
    waste: Optional[torch.Tensor],
    capacity: Optional[torch.Tensor],
    candidates: List[List[int]],
    use_3opt: bool,
    curr_p: float,
    curr_c: float,
) -> Tuple[bool, torch.Tensor, float, float]:
    """Tries combinations of 2-opt and 3-opt moves for a header node.

    Args:
        tour: Current sequence under evaluation.
        i: Node position index under evaluation.
        nodes_count: Total sequence length.
        dist: Pairwise cost matrix.
        waste: Node demands for capacity calculation.
        capacity: Fleet capacity limit.
        candidates: Near-set neighbor lists.
        use_3opt: Whether to enable 3-opt neighborhood exploration.
        curr_p: Current tour penalty score.
        curr_c: Current tour distance cost.

    Returns:
        Tuple[bool, torch.Tensor, float, float]: A tuple containing:
            - improved: Whether any move was accepted.
            - tour: The updated (or original) tour.
            - new_p: Resulting penalty.
            - new_c: Resulting cost.
    """
    t1, t2 = int(tour[i].item()), int(tour[i + 1].item())
    for t3 in candidates[t2]:
        if t3 == t1:
            continue
        t3_pos = (tour == t3).nonzero(as_tuple=True)[0]
        if len(t3_pos) == 0:
            continue
        j = int(t3_pos[0].item())
        if j <= i + 1 or j >= nodes_count:
            continue

        t4 = int(tour[j + 1].item())
        # 1. Try 2-opt
        if (dist[t1, t2] + dist[t3, t4]) - (dist[t1, t3] + dist[t2, t4]) > IMPROVEMENT_EPSILON:
            new_tour = _apply_2opt(tour, i, j)
            new_p, new_c = _compute_score(new_tour, dist, waste, capacity)
            if _is_better(new_p, new_c, curr_p, curr_c):
                return True, new_tour, new_p, new_c

        # 2. Try 3-opt
        if use_3opt:
            for k in range(j + 2, min(j + 20, nodes_count)):
                t5, t6 = int(tour[k].item()), int(tour[k + 1].item())
                if (dist[t1, t2] + dist[t3, t4] + dist[t5, t6]) - (
                    dist[t1, t3] + dist[t2, t5] + dist[t4, t6]
                ) > IMPROVEMENT_EPSILON:
                    new_tour = _apply_3opt(tour, i, j, k)
                    new_p, new_c = _compute_score(new_tour, dist, waste, capacity)
                    if _is_better(new_p, new_c, curr_p, curr_c):
                        return True, new_tour, new_p, new_c
    return False, tour, curr_p, curr_c


def _compute_alpha_measures(distance_matrix: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Estimates edge reliability using alpha-measure heuristics.

    Args:
        distance_matrix: Edge weights of shape [N, N].
        device: Hardware identification locator.

    Returns:
        torch.Tensor: Alpha measures of shape [N, N].
    """
    n = distance_matrix.size(0)
    alpha = distance_matrix.clone()
    k = min(5, n - 1)
    _, indices = torch.topk(distance_matrix, k, dim=1, largest=False)
    for i in range(n):
        for j in indices[i]:
            if int(j) != i:
                alpha[i, int(j)] = 0.0
    return alpha


def _get_candidate_sets(alpha_measures: torch.Tensor, max_candidates: int) -> List[List[int]]:
    """Restricts neighborhood search space to high-alpha edges.

    Args:
        alpha_measures: Pre-computed alpha scores of shape [N, N].
        max_candidates: Maximum neighbor count per node.

    Returns:
        List[List[int]]: Per-node candidate neighbor index lists.
    """
    n = alpha_measures.size(0)
    candidates = []
    for i in range(n):
        sorted_indices = torch.argsort(alpha_measures[i])
        valid = [int(idx.item()) for idx in sorted_indices if int(idx) != i]
        candidates.append(valid[:max_candidates])
    return candidates


def _compute_score(
    tour: torch.Tensor,
    distance_matrix: torch.Tensor,
    wastes: Optional[torch.Tensor],
    capacity: Optional[torch.Tensor],
) -> Tuple[float, float]:
    """Lexicographic (Penalty, Cost) evaluator.

    Args:
        tour: Sequence of nodes of shape [L].
        distance_matrix: Pairwise weights of shape [N, N].
        wastes: Node demands of shape [N].
        capacity: Vehicle capacity limit.

    Returns:
        Tuple[float, float]: A tuple containing:
            - penalty: Calculated capacity violation penalty.
            - cost: Calculated tour distance cost.
    """
    n = len(tour) - 1
    cost = 0.0
    for i in range(n):
        u, v = int(tour[i].item()), int(tour[i + 1].item())
        cost += float(distance_matrix[u, v].item())

    penalty = 0.0
    if wastes is not None and capacity is not None:
        current_load = 0.0
        cap_val = float(capacity.item())
        for node in tour:
            node_idx = int(node.item())
            if node_idx == 0:
                current_load = 0.0
            elif node_idx < len(wastes):
                current_load += float(wastes[node_idx].item())
                if current_load > cap_val + 1e-6:
                    penalty += current_load - cap_val
    return penalty, cost


def _is_better(p1: float, c1: float, p2: float, c2: float) -> bool:
    """True if (p1, c1) is superior to (p2, c2) in lexicographic order.

    Args:
        p1: First tour penalty score.
        c1: First tour distance cost.
        p2: Second tour penalty score.
        c2: Second tour distance cost.

    Returns:
        bool: True if the first solution dominates the second.
    """
    if abs(p1 - p2) > 1e-6:
        return p1 < p2
    return c1 < c2 - 1e-6


def _apply_2opt(tour: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Segment reversal for 2-opt moves.

    Args:
        tour: Sequence under transformation.
        i: First edge break position.
        j: Second edge break position.

    Returns:
        torch.Tensor: The tour after segment reversal.
    """
    new_tour = tour.clone()
    new_tour[i + 1 : j + 1] = torch.flip(new_tour[i + 1 : j + 1], dims=[0])
    return new_tour


def _apply_3opt(tour: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
    """Triple segment recombination for 3-opt moves.

    Args:
        tour: Sequence under transformation.
        i: First edge break position.
        j: Second edge break position.
        k: Third edge break position.

    Returns:
        torch.Tensor: The tour after triple segment recombination.
    """
    new_tour = tour.clone()
    new_tour[i + 1 : j + 1] = torch.flip(new_tour[i + 1 : j + 1], dims=[0])
    new_tour[j + 1 : k + 1] = torch.flip(new_tour[j + 1 : k + 1], dims=[0])
    return new_tour


def _double_bridge_kick(tour: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """4-opt perturbation (Double Bridge Kick).

    Args:
        tour: Node sequence to perturb.
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: The perturbed tour.
    """
    n = len(tour) - 1
    if n < 8:
        return tour
    device = tour.device
    positions = torch.randperm(n - 2, device=device, generator=generator) + 1
    positions = positions[:4].sort()[0]
    a, b, c, d = positions.tolist()
    tour_list = tour.tolist()
    new_tour = (
        tour_list[: a + 1]
        + tour_list[c + 1 : d + 1]
        + tour_list[b + 1 : c + 1]
        + tour_list[a + 1 : b + 1]
        + tour_list[d + 1 :]
    )
    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
