"""
Distance-Based Particle Swarm Optimization for VRPP.

Rigorous implementation replacing metaphor-based "Firefly Algorithm".

Attributes:
    DistancePSOSolver: PSO solver with inertia-weighted velocity updates.
    DistancePSOConfig: Configuration parameters dataclass.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_distance_based_algorithm import DistancePSOSolver, DistancePSOConfig
    >>> params = DistancePSOConfig(population_size=20, max_iterations=500)
    >>> solver = DistancePSOSolver(dist_matrix, wastes, capacity, R, C, params)
    >>> routes, profit, cost = solver.solve()
"""

from logic.src.configs.policies.psoda import DistancePSOConfig

from .solver import DistancePSOSolver

__all__ = ["DistancePSOSolver", "DistancePSOConfig"]
