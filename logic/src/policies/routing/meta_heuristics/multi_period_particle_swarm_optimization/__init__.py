"""
Multi-Period Particle Swarm Optimization (MP-PSO) matheuristic package.

Provides a multi-period particle swarm optimization approach for complex routing
problems with temporal constraints.

Attributes:
    MultiPeriodPSOPolicy: Policy class for the MP-PSO approach.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.multi_period_particle_swarm_optimization import MultiPeriodPSOPolicy
"""

from .policy_mp_pso import MultiPeriodPSOPolicy

__all__ = ["MultiPeriodPSOPolicy"]
