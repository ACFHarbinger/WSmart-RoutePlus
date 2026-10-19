"""
Route improvement utilities.
Includes route decoding, efficiency calculation, and lightweight optimization.

Attributes:
    EfficiencyOptimizer: Optimizes routes as a route improvement step.
    decode_routes: Converts action sequences to clean route lists on CPU.
    calculate_efficiency: Computes kg/km efficiency metric.

Example:
    >>> from logic.src.pipeline.learning.common.route_improvement import EfficiencyOptimizer, decode_routes, calculate_efficiency
    >>> optimizer = EfficiencyOptimizer(problem)
    >>> routes = optimizer.optimize(actions, num_nodes)
    >>> efficiency = calculate_efficiency(routes, dist_matrix, waste, capacity)
    >>> print(efficiency)
"""

from typing import List

import torch


class EfficiencyOptimizer:
    """Optimizes routes as a route improvement step.

    Attributes:
        problem: Problem description.
    """

    def __init__(self, problem, **kwargs):
        """
        Initialize EfficiencyOptimizer.

        Args:
            problem: Problem description.
            kwargs: Additional keyword arguments.
        """
        self.problem = problem

    def optimize(self, routes: List[torch.Tensor], **kwargs):
        """Refine routes using local search or heuristics.

        Args:
            routes: List of routes to optimize.
            kwargs: Additional keyword arguments.

        Returns:
            Optimized routes.
        """
        # This matches the legacy EfficiencyOptimizer which might run 2-opt
        # We can link to our new vectorized policies here if needed
        return routes


def decode_routes(actions: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Convert action sequences to clean route lists on CPU.

    Args:
        actions: Tensor of action sequences.
        num_nodes: Number of nodes in each route.

    Returns:
        List of decoded routes.
    """
    routes = []
    actions_cpu = actions.cpu()
    for action in actions_cpu:
        # Cut at 0s
        route = action[action != 0].tolist()
        routes.append(route)
    return routes


def calculate_efficiency(routes, dist_matrix, waste, capacity):
    """
    Compute kg/km efficiency metric.
    Efficiency = Total Waste Collected / Total Distance Traveled.
    Args:
        routes: List of routes.
        dist_matrix: Distance matrix.
        waste: Waste at each node.
        capacity: Vehicle capacity.

    Returns:
        Efficiency metric.
    """
    total_waste = 0.0
    total_dist = 0.0
    for route in routes:
        if not route:
            continue

        # 1. Calculate Waste
        route_tensor = torch.tensor(route, dtype=torch.long)

        # Gather waste
        # wastes needs to be accessible by index
        waste = waste[route_tensor].sum().item()
        total_waste += waste

        # 2. Calculate Distance
        # Depot -> First Node
        d = dist_matrix[0, route[0]].item()

        # Inter-node
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            d += dist_matrix[u, v].item()

        # Last Node -> Depot
        d += dist_matrix[route[-1], 0].item()

        total_dist += d

    if total_dist < 1e-6:
        return 0.0

    return total_waste / total_dist
