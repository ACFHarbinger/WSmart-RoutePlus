"""
Multi-Period Particle Swarm Optimization (MP-PSO) matheuristic package.

Provides a multi-period particle swarm optimization approach for complex routing
problems with temporal constraints.

Attributes:
    MultiPeriodPSOPolicy: Policy class for the MP-PSO approach.

"""

from .policy_mp_pso import MultiPeriodPSOPolicy

__all__ = ["MultiPeriodPSOPolicy"]
