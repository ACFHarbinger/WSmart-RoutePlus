"""
POPMUSIC (Partial Optimization Metaheuristic Under Special Intensification Conditions) Policy.

This matheuristic framework decomposes a large problem into overlapping subproblems
and optimizes them iteratively.

Attributes:
    POPMUSICPolicy (POPMUSICPolicy): The POPMUSIC policy class.
    run_popmusic (callable): The run_popmusic function.

Example:
    >>> from logic.src.policies.routing.matheuristics.
    ...     partial_optimization_metaheuristic_under_special_intensification_conditions import run_popmusic
    >>>
    >>> run_popmusic(cfg)
"""

from .policy_popmusic import POPMUSICPolicy
from .solver import run_popmusic

__all__ = ["POPMUSICPolicy", "run_popmusic"]
