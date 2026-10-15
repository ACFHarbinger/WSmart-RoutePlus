"""
Particle Swarm Optimization Memetic Algorithm (PSOMA) for VRPP.

Attributes:
    PSOMASolver: Core PSOMA solver class.
    PSOMAParticle: Particle representation for the swarm.
    PSOMAPolicy: Policy adapter for PSOMA.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.particle_swarm_optimization_memetic_algorithm import PSOMASolver
    >>> solver = PSOMASolver(dist_matrix, wastes, capacity, R, C)
    >>> routes, profit, cost = solver.solve()
"""

from .particle import PSOMAParticle
from .policy_psoma import PSOMAPolicy
from .solver import PSOMASolver

__all__ = ["PSOMASolver", "PSOMAParticle", "PSOMAPolicy"]
