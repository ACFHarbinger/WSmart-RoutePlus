"""Particle Swarm Optimization (PSO) with velocity momentum.

Attributes:
    PSOConfig: Parameter dataclass for Particle Swarm Optimization.
    PSOPolicyAdapter: Policy class for Particle Swarm Optimization.
    PSOSolver: Main solver class for Particle Swarm Optimization.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.particle_swarm_optimization import PSOPolicyAdapter
    >>> policy = PSOPolicyAdapter()
"""

from logic.src.configs.policies.routing.meta_heuristics.pso import PSOConfig

from .policy_pso import PSOPolicyAdapter
from .solver import PSOSolver

__all__ = ["PSOConfig", "PSOPolicyAdapter", "PSOSolver"]
