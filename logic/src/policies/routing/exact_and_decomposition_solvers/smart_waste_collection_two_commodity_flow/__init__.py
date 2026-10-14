r"""SWC-TCF (Smart Waste Collection - Two-Commodity Flow) Adapter Package.

Attributes:
    run_swc_tcf_optimizer: High-level dispatcher for TCF solvers.
    SWCTCFPolicy: Simulator policy adapter for TCF.
    _run_gurobi_optimizer: Native Gurobi TCF implementation.

Example:
    >>> from logic.src.policies.routing.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow import SWCTCFPolicy
    >>> policy = SWCTCFPolicy()
"""

from .dispatcher import run_swc_tcf_optimizer
from .gurobi import _run_gurobi_optimizer
from .policy_swc_tcf import SWCTCFPolicy

__all__ = [
    "_run_gurobi_optimizer",
    "run_swc_tcf_optimizer",
    "SWCTCFPolicy",
]
