"""Multi-Stage Branch-and-Price-and-Cut with Set Partitioning package.

Exposes the main runner function and individual engines.

Attributes:
    run_ms_bpc_sp (function): Entry point for the MS BPC SP solver engine.
    policy_ms_bpc_sp (module): Policy wrapper for MS BPC SP.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition import run_ms_bpc_sp
    >>> result = run_ms_bpc_sp(problem_instance, params)
"""

from . import policy_ms_bpc_sp
from .ms_bpc_sp_engine import run_ms_bpc_sp

__all__ = ["run_ms_bpc_sp", "policy_ms_bpc_sp"]
