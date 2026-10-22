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
    60.0
    >>> config.seed
    None
    >>> config.engine
    'gurobi'
    >>> config.framework
    'gurobi'
    >>> config.packing
    None
    >>> config.route_improvement
    None
"""

from dataclasses import dataclass
from typing import List, Optional

from logic.src.configs.policies.packing.dispatch import PackingConfig

from logic.src.configs.policies.route_improvement import RouteImprovingConfig


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
        framework: Solver framework to use ('gurobi'/'native', 'ortools', 'pyomo').
        formulation: Native Gurobi model: 'paper' (published SWCR model, default) or 'directed'.
        depot_inflow: 'equal' (eq. 15, default) or 'le' (collectioncompare notebook).
        solver_tuning: Optional Gurobi tuning (MIPFocus, cuts, 1% gap); off by default.
        relax_forced_on_infeasible: Optional re-solve without forced visits; off by default.
        link_depot_arcs: Optional x[0, j] <= g[j]; off by default.
        max_arc_distance_km: Optional arc length cutoff; None keeps every arc.
        warm_start: Native Gurobi only; off by default. Loads Clarke-Wright MIP
            starts, which are not part of the published model (#41).
        asymmetric: Directed arcs at their own cost (not the undirected published
            model); requires formulation 'paper' and cannot be combined with warm_start.
        no_solution_time_factor: A day that reaches the time limit with no incumbent
            is re-solved once, at the default tolerance, with this multiple of the
            limit (at least 1).
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
    """

    Omega: float = 0.1
    psi: float = 1.0
    time_limit: float = 60.0
    seed: Optional[int] = None
    engine: str = "gurobi"
    gurobi_threads: int = 2
    gurobi_soft_mem_limit_gb: float = 5.0
    framework: str = "gurobi"
    formulation: str = "paper"
    depot_inflow: str = "equal"
    solver_tuning: bool = False
    relax_forced_on_infeasible: bool = False
    link_depot_arcs: bool = False
    max_arc_distance_km: Optional[float] = None
    warm_start: bool = False
    asymmetric: bool = False
    no_solution_time_factor: float = 3.0
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
