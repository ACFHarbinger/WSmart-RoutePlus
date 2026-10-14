r"""Logic-Based Benders Decomposition (LBBD) package.

This module provides the core implementation of the LBBD solver, which
decomposes the Integrated Routing and Inventory Management (IRIM) problem
into an assignment master problem and routing subproblems.

Attributes:
    LBBDEngine: Core solver engine coordinating master and subproblems.
    LBBDPolicy: Policy interface for the WSmart+ simulator.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.logic_based_benders_decomposition import LBBDPolicy
    >>> policy = LBBDPolicy()
"""

from . import policy_lbbd as policy_lbbd
from .lbbd_engine import LBBDEngine
from .policy_lbbd import LBBDPolicy

__all__ = ["policy_lbbd", "LBBDEngine", "LBBDPolicy"]
