"""
ILS-RVND-SP (Iterated Local Search - Randomized Variable Neighborhood Descent - Set Partitioning) configuration dataclasses.
Attributes:
    ILSRVNDSPConfig: Configuration for the ILS-RVND-SP algorithm.

Example:
    >>> from configs.policies.ils_rvnd_sp import ILSRVNDSPConfig
    >>> config = ILSRVNDSPConfig()
    >>> config.time_limit
    120.0
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig

from .abc import ABCConfig


@dataclass
class ILSRVNDSPConfig(ABCConfig):
    """Configuration for the ILS-RVND-SP algorithm.

    Attributes:
        max_restarts (int): Maximum number of restarts for ILS.
        max_iter_ils (int): Maximum number of iterations for ILS.
        perturbation_strength (int): Strength of perturbation.
        use_set_partitioning (bool): Whether to use Set Partitioning.
        mip_time_limit (float): Time limit for MIP solver.
        sp_mip_gap (float): Gap tolerance for MIP solver.
        N (int): Parameter N.
        A (float): Parameter A.
        MaxIter_a (int): Maximum iterations for phase a.
        MaxIter_b (int): Maximum iterations for phase b.
        MaxIterILS_b (int): Maximum iterations for ILS in phase b.
        TDev_a (float): Temperature deviation for phase a.
        TDev_b (float): Temperature deviation for phase b.
        vrpp (bool): Whether the problem is a VRRP.
        profit_aware_operators (bool): Whether to use profit-aware operators.
        time_limit (float): Time limit for the algorithm.
        seed (Optional[int]): Seed for the random number generator.
        engine (str): Solver engine to use.
        framework (str): Framework for the algorithm.
        local_search_iterations (int): Number of local search iterations.
        packing (Optional[List[PackingConfig]]): Packing configurations.
        route_improvement (Optional[List[RouteImprovingConfig]]): Route improvement configurations.
        acceptance_criterion (AcceptanceConfig): Acceptance criterion configuration.
    """

    # ILS Phase parameters
    max_restarts: int = 10
    max_iter_ils: int = 50
    perturbation_strength: int = 2

    # Set Partitioning (SP) Phase parameters
    use_set_partitioning: bool = True
    mip_time_limit: float = 60.0
    sp_mip_gap: float = 0.01

    # Paper-specific advanced parameters
    N: int = 150
    A: float = 11.0
    MaxIter_a: int = 50
    MaxIter_b: int = 100
    MaxIterILS_b: int = 2000
    TDev_a: float = 0.05
    TDev_b: float = 0.005

    # Global parameters
    vrpp: bool = True
    profit_aware_operators: bool = False
    time_limit: float = 120.0
    seed: Optional[int] = None
    engine: str = "gurobi"
    framework: str = "ortools"
    local_search_iterations: int = 500

    # --- Infrastructure Hooks ---
    packing: Optional[List[Any]] = field(default_factory=list)
    route_improvement: Optional[List[Any]] = field(default_factory=list)
    acceptance_criterion: AcceptanceConfig = field(default_factory=lambda: AcceptanceConfig(method="only_improving"))
    max_vehicles: int = 0
