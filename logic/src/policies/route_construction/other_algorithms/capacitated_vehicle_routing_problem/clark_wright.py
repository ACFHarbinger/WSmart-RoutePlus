"""
Internal implementation of Clark-Wright Savings algorithm for CVRP.

Attributes:
    clarke_wright_solve(dist_matrix, wastes, capacity, to_collect, depot=0)
        Solve CVRP using Clark-Wright Savings heuristic.

        Args:
            dist_matrix: Distance matrix.
            wastes: Waste demands.
            capacity: Vehicle capacity.
            to_collect: Nodes to collect.
            depot: Depot node.

        Returns:
            List of routes.

Example:
    >>> from logic.src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem import clarke_wright_solve
    >>> routes = clarke_wright_solve(dist_matrix, wastes, capacity, to_collect)
"""

from typing import Dict, List

import numpy as np


def clarke_wright_solve(
    dist_matrix: np.ndarray, wastes: Dict[int, float], capacity: float, to_collect: List[int], depot: int = 0
) -> List[int]:
    """
    Solve CVRP using Clark-Wright Savings heuristic.
    Args:
        dist_matrix: Description of dist_matrix.
        wastes: Description of wastes.
        capacity: Description of capacity.
        to_collect: Description of to_collect.
        depot: Description of depot.

    Returns:
        Description of return value.
    """
    to_collect = [int(node) for node in to_collect]
    if not to_collect:
        return [depot]

    # 1. Initialize with one-trip-per-node routes
    routes = [[node] for node in to_collect]

    # 2. Compute savings: s_ij = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in range(len(to_collect)):
        for j in range(i + 1, len(to_collect)):
            ni, nj = to_collect[i], to_collect[j]
            s = dist_matrix[depot, ni] + dist_matrix[depot, nj] - dist_matrix[ni, nj]
            savings.append((s, ni, nj))

    savings.sort(key=lambda x: x[0], reverse=True)

    # 3. Merge routes
    node_to_route = {node: i for i, node in enumerate(to_collect)}
    route_loads = [wastes.get(node, 0.0) for node in to_collect]

    for _s, ni, nj in savings:
        ri_idx = node_to_route[ni]
        rj_idx = node_to_route[nj]

        if ri_idx == rj_idx:
            continue

        ri = routes[ri_idx]
        rj = routes[rj_idx]

        # Check if they can be merged (must be interior vs exterior)
        # For simplicity, check if ni is end of ri and nj is start of rj
        can_merge = False
        new_route = []
        if ri[-1] == ni and rj[0] == nj:
            can_merge = True
            new_route = ri + rj
        elif ri[0] == ni and rj[-1] == nj:
            can_merge = True
            new_route = rj + ri
        elif ri[-1] == ni and rj[-1] == nj:
            can_merge = True
            new_route = ri + rj[::-1]
        elif ri[0] == ni and rj[0] == nj:
            can_merge = True
            new_route = ri[::-1] + rj

        if can_merge:
            new_load = route_loads[ri_idx] + route_loads[rj_idx]
            if new_load <= capacity:
                # Merge!
                routes[ri_idx] = new_route
                route_loads[ri_idx] = new_load
                for node in rj:
                    node_to_route[node] = ri_idx
                routes[rj_idx] = []  # Mark as merged

    # 4. Flatten and Return
    final_tour = [depot]
    for r in routes:
        if r:
            final_tour.extend(r)
            final_tour.append(depot)

    return final_tour
