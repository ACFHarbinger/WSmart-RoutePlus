"""Tabu Search (TS) module.

Attributes:
    TSParams: Parameter dataclass for the Tabu Search.
    TSPolicy: Policy class for Tabu Search.
    TSSolver: Main solver class for Tabu Search.

Example:
    >>> from logic.src.policies.routing.meta_heuristics.tabu_search import TSPolicy
    >>> policy = TSPolicy()
"""

from .params import TSParams
from .policy_ts import TSPolicy
from .solver import TSSolver

__all__ = ["TSParams", "TSPolicy", "TSSolver"]
