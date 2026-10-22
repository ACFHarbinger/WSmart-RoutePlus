"""Ejection chain operator.

This module provides a GPU-accelerated implementation of the Ejection chain
operator, a sophisticated fleet minimization heuristic that attempts to
empty a route by ejecting its customers into other routes.

Attributes:
    vectorized_ejection_chain: Attempts to eliminate routes by ejecting their nodes into other routes.

Example:
    >>> from logic.src.policies.vectorized.routing.operators.exchange.ejection_chain import vectorized_ejection_chain
    >>> tours = vectorized_ejection_chain(tours, dist_matrix, capacities, wastes, max_depth)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def vectorized_ejection_chain(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    wastes: Optional[torch.Tensor] = None,
    max_depth: int = 5,
    target_route_reduction: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized ejection chain operator for fleet minimization.

    Identifies under-utilized routes and attempts to eliminate them by
    redistributing their nodes into other routes. If direct insertion
    is blocked by capacity, it triggers a chain of displacements.

    Args:
        tours: Batch of node sequences of shape [B, N].
        distance_matrix: Edge cost tensor of shape [B, N+1, N+1] or [N+1, N+1].
        capacities: Vehicle capacity per instance of shape [B] or scalar.
        wastes: Node demand metadata of shape [B, N+1] or [N+1].
        max_depth: Recursion limit for the ejection sequence.
        target_route_reduction: Limit for how many routes to attempt to empty.
        generator: Torch device-side RNG.

    Returns:
        torch.Tensor: Modified tours of shape [B, N].
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

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    # wastes are required for ejection chain
    if wastes is None:
        # Cannot perform ejection chain without waste information
        return tours if is_batch else tours.squeeze(0)

    # Handle wastes
    if wastes.dim() == 1:
        wastes = wastes.unsqueeze(0).expand(B, -1)

    # Handle capacities
    if capacities is None:
        capacities = torch.full((B,), float("inf"), device=device)
    elif capacities.dim() == 0:
        capacities = capacities.unsqueeze(0).expand(B)

    # Process each batch instance separately (ejection chain is inherently sequential)
    for b in range(B):
        tour = tours[b]
        dist = distance_matrix[b]
        waste = wastes[b]
        capacity = capacities[b]

        # Extract routes and calculate loads
        routes = _get_routes_with_loads(tour, waste, N)
        if len(routes) < 2:
            continue

        # Try to eliminate routes
        routes_to_eliminate = target_route_reduction if target_route_reduction else len(routes) - 1
        routes_eliminated = 0

        # Sort routes by load (ascending)
        route_indices = sorted(range(len(routes)), key=lambda i: routes[i][3])

        for idx in route_indices:
            if routes_eliminated >= routes_to_eliminate:
                break

            success, modified_tour = _attempt_route_elimination(
                tour=tour,
                route_data=routes[idx],
                dist=dist,
                waste=waste,
                capacity=capacity,
                max_depth=max_depth,
                depot_positions=torch.where(tour == 0)[0].tolist(),
            )

            if success:
                tour = modified_tour
                routes_eliminated += 1
                routes = _get_routes_with_loads(tour, waste, N)

        tours[b] = tour

    return tours if is_batch else tours.squeeze(0)


def _try_insert_with_ejection_chain(
    tour: torch.Tensor,
    node: int,
    source_start: int,
    source_end: int,
    dist: torch.Tensor,
    waste: torch.Tensor,
    capacity: torch.Tensor,
    max_depth: int,
    ejection_log: List[Tuple[int, int]],
    depot_positions: List[int],
) -> bool:
    """Attempts node insertion, exploring chain displacements if capacity-limited.

    Args:
        tour: Individual sequence metadata of shape [N].
        node: Customer index to insert.
        source_start: Start index of the source route.
        source_end: End index of the source route.
        dist: Instance distance matrix of shape [N+1, N+1].
        waste: Node demand metadata of shape [N+1].
        capacity: Vehicle capacity limit.
        max_depth: Remaining recursion depth.
        ejection_log: List tracker for successful movements.
        depot_positions: Index positions of all depots in sequence.

    Returns:
        bool: True if insertion was successful.
    """
    if max_depth <= 0:
        return False

    device = tour.device
    node_waste = waste[node].item()

    # Try direct insertion into each route
    for i in range(len(depot_positions) - 1):
        route_start = int(depot_positions[i] + 1)
        route_end = int(depot_positions[i + 1])

        # Skip source route
        if route_start == source_start:
            continue

        # Get route nodes
        route_nodes = tour[route_start:route_end]
        route_nodes = route_nodes[route_nodes > 0]

        # Check capacity
        route_load = sum(waste[n].item() for n in route_nodes)

        if route_load + node_waste <= capacity.item():
            # Find best insertion position
            best_pos = _find_best_insertion_in_route(tour, node, route_start, route_end, dist)

            # Insert node at best position
            tour_list = tour.tolist()
            tour_list.insert(best_pos, node)
            tour[:] = torch.tensor(tour_list[: len(tour)], device=device)

            ejection_log.append((node, best_pos))
            return True

    return False


def _get_routes_with_loads(tour: torch.Tensor, waste: torch.Tensor, N: int) -> List[Tuple[int, int, List[int], float]]:
    """Calculates loading metrics for all routes in the current tour.

    Args:
        tour: Sequence metadata of shape [N].
        waste: Instance demand metadata of shape [N+1].
        N: System problem size (nodes).

    Returns:
        List[Tuple[int, int, List[int], float]]: Detailed per-route metrics:
            (start_idx, end_idx, list_of_nodes, total_load).
    """
    depot_positions = torch.where(tour == 0)[0].tolist()
    routes = []
    for i in range(len(depot_positions) - 1):
        start = int(depot_positions[i] + 1)
        end = int(depot_positions[i + 1])
        if end > start:
            nodes = [int(n) for n in tour[start:end].tolist() if 0 < n < N]
            if nodes:
                load = float(sum(waste[n].item() for n in nodes))
                routes.append((start, end, nodes, load))
    return routes


def _attempt_route_elimination(
    tour: torch.Tensor,
    route_data: Tuple[int, int, List[int], float],
    dist: torch.Tensor,
    waste: torch.Tensor,
    capacity: torch.Tensor,
    max_depth: int,
    depot_positions: List[int],
) -> Tuple[bool, torch.Tensor]:
    """Attempts to empty an entire route into the remaining fleet.

    Args:
        tour: Source sequence metadata of shape [N].
        route_data: Metadata for the target route to eliminate.
        dist: Pairwise weights of shape [N+1, N+1].
        waste: Node demands of shape [N+1].
        capacity: Fleet limits.
        max_depth: Recursion limit for chain search.
        depot_positions: Depot marker locations in the tour.

    Returns:
        Tuple[bool, torch.Tensor]: A tuple containing:
            - success: True if the route was successfully cleared.
            - modified_tour: The resulting updated tour sequence.
    """
    start, end, nodes, _ = route_data
    ejection_log: List[Tuple[int, int]] = []
    success = True
    modified_tour = tour.clone()

    for node in nodes:
        if not _try_insert_with_ejection_chain(
            modified_tour,
            node,
            start,
            end,
            dist,
            waste,
            capacity,
            max_depth,
            ejection_log,
            depot_positions,
        ):
            success = False
            break

    if success:
        modified_tour = modified_tour.clone()
        # Mark nodes in the original route as removed (using -1 to avoid depot confusion)
        modified_tour[start:end] = -1
        return True, modified_tour
    return False, tour


def _find_best_insertion_in_route(tour: torch.Tensor, node: int, start: int, end: int, dist: torch.Tensor) -> int:
    """Uses greedy cost evaluation to find the optimal insertion index.

    Args:
        tour: Sequence metadata of shape [N].
        node: Node to evaluate for insertion.
        start: Route region start.
        end: Route region end.
        dist: Pairwise distance weights.

    Returns:
        int: Index position providing the lowest cost insertion.
    """
    best_cost = float("inf")
    best_pos = start
    for pos in range(start, end + 1):
        prev = int(tour[pos - 1].item()) if pos > start else 0
        nxt = int(tour[pos].item()) if pos < end else 0
        cost = float(dist[prev, node].item() + dist[node, nxt].item())
        if nxt > 0:
            cost -= float(dist[prev, nxt].item())
        if cost < best_cost:
            best_cost, best_pos = cost, pos
    return best_pos
