"""
SWC-TCF (Smart Waste Collection - Two-Commodity Flow) policy configuration.

Attributes:
    SWCTCFConfig: Configuration for the Smart Waste Collection - Two-Commodity Flow (SWC-TCF) policy.

Example:
    >>> from configs.policies.swc_tcf import SWCTCFConfig
    >>> config = SWCTCFConfig()
    >>> config.Omega
    0.1
    >>> config.psi
    1.0
    >>> config.time_limit
    600.0
    >>> config.seed
    None
    >>> config.engine
    'gurobi'
    >>> config.framework
    'ortools'
    >>> config.mandatory_selection
    None
    >>> config.route_improvement
    None
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class SWCTCFConfig:
    """Configuration for Smart Waste Collection - Two-Commodity Flow (SWC-TCF) policy.

    Attributes:
        Omega: Fixed cost per vehicle used (EUR).
        psi: Fill fraction at or above which a bin is forced into the plan.
        gurobi_threads: Native Gurobi threads per solve (positive; independent of simulator workers).
        gurobi_soft_mem_limit_gb: Native Gurobi soft memory ceiling in decimal GB.
        time_limit: Maximum time in seconds for the solver.
        engine: Solver engine to use ('gurobi', 'scip', 'highs', or 'cplex').
        framework: Solver framework to use ('ortools', 'pyomo').
        warm_start: Native Gurobi only; off by default. Loads Clarke-Wright MIP
            starts, which are not part of the published model (#41).
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
    """

    Omega: float = 0.1
    psi: float = 1.0
    time_limit: float = 600.0
    seed: Optional[int] = None
    engine: str = "gurobi"
    gurobi_threads: int = 2
    gurobi_soft_mem_limit_gb: float = 5.0
    framework: str = "ortools"
    warm_start: bool = False
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
