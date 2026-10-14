"""Branch-and-Price-and-Cut package.

Exposes the main runner function and individual engines.

Attributes:
    run_bpc (function): Entry point for the BPC solver engine.
    policy_bpc (module): Policy wrapper for BPC.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut import run_bpc
    >>> result = run_bpc(problem_instance, params)
"""

from . import policy_bpc
from .bpc_engine import run_bpc

__all__ = ["run_bpc", "policy_bpc"]
