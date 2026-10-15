"""
Multi-Period Iterated Local Search (MP-ILS) matheuristic package.

Provides a multi-period iterated local search approach for complex routing
problems with temporal constraints.

Attributes:
    MultiPeriodILSPolicy: Policy class for the MP-ILS approach.

"""

from .policy_mp_ils import MultiPeriodILSPolicy

__all__ = ["MultiPeriodILSPolicy"]
