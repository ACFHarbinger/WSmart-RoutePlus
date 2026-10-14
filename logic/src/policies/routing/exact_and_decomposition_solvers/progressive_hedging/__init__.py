r"""Progressive Hedging (PH) policy implementation.

Horizontal decomposition for stochastic routing problems.

Attributes:
    ProgressiveHedgingEngine: Core solver engine for PH decomposition.
    ProgressiveHedgingPolicy: Policy adapter for the WSmart+ simulator.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.progressive_hedging import ProgressiveHedgingPolicy
    >>> policy = ProgressiveHedgingPolicy()
"""

from .ph_engine import ProgressiveHedgingEngine
from .policy_ph import ProgressiveHedgingPolicy

__all__ = ["ProgressiveHedgingEngine", "ProgressiveHedgingPolicy"]
