"""Adaptive Branch-and-Price-and-Cut with Heuristic Guidance (ABPC-HG) configuration.

Attributes:
   ABPCHGConfig: ABPC-HG policy configuration.

Example:
    >>> from logic.src.configs.policies import ABPCHGConfig
    >>> config = ABPCHGConfig()
    >>> print(config)
    ABPCHGConfig(gamma=0.95, seed=None)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ABPCHGConfig:
    """Config for Adaptive Branch-and-Price-and-Cut with Heuristic Guidance (ABPC-HG).

    Attributes:
        gamma (float): Inter-day discount factor.
        seed (Optional[int]): Random seed.
        overflow_penalty (float): Revenue multiplier for projected overflows.
        ph_base_rho (float): Base penalty for PH consensus.
        ph_max_iterations (int): Max PH iterations.
        ph_convergence_tol (float): PH convergence tolerance.
        alns_iterations (int): Max ALNS cycles per pricing call.
        alns_max_routes (int): Max routes returned per ALNS call.
        alns_rc_tolerance (float): Reduced-cost threshold for ALNS.
        alns_remove_fraction (float): Fraction of nodes removed in destroy step.
        dive_penalty_M (float): Big-M penalty for diving infeasibility.
        fo_tabu_length (int): Tabu list length for Fix-and-Optimize.
        fo_max_unfix (int): Max bins unfixed in F&O corridor.
        fo_strategy (str): Cluster selection strategy for F&O.
        fo_max_iterations (int): Max F&O passes.
        ml_reliability_c (float): Reliability branching blending coefficient.
        ml_pseudocost_ema_alpha (float): Alpha for pseudo-cost EMA.
        sc_consensus_threshold (float): Base threshold for scenario consensus.
        benders_max_iterations (int): Max Benders master-subproblem cycles.
        benders_convergence_tol (float): Gap tolerance for Benders termination.
        benders_cut_pool_max (int): Max Benders cuts in pool.
        max_visits_per_bin (int): Max visits per bin over horizon.
        theta_upper_bound (float): Initial UB for theta in master MIP.
        gurobi_master_time_limit (float): Gurobi time limit for master (s).
        gurobi_sub_time_limit (float): Gurobi time limit for subproblem (s).
        gurobi_mip_gap (float): Gurobi relative MIP gap.
        gurobi_output_flag (bool): Enable Gurobi console output.
        subproblem_relax (bool): Solve subproblems as LPs (True) or MIPs (False).
    """

    gamma: float = 0.95
    seed: Optional[int] = None
    overflow_penalty: float = 2.0
    ph_base_rho: float = 1.0
    ph_max_iterations: int = 100
    ph_convergence_tol: float = 1e-4
    alns_iterations: int = 50
    alns_max_routes: int = 5
    alns_rc_tolerance: float = 1e-4
    alns_remove_fraction: float = 0.25
    dive_penalty_M: float = 10_000.0
    fo_tabu_length: int = 10
    fo_max_unfix: int = 5
    fo_strategy: Literal["overflow_urgency", "scenario_divergence"] = "overflow_urgency"
    fo_max_iterations: int = 20
    ml_reliability_c: float = 1.0
    ml_pseudocost_ema_alpha: float = 0.5
    sc_consensus_threshold: float = 0.95
    benders_max_iterations: int = 50
    benders_convergence_tol: float = 1e-3
    benders_cut_pool_max: int = 500
    max_visits_per_bin: int = 1
    theta_upper_bound: float = 1e6
    gurobi_master_time_limit: float = 60.0
    gurobi_sub_time_limit: float = 30.0
    gurobi_mip_gap: float = 1e-4
    gurobi_output_flag: bool = False
    subproblem_relax: bool = True


    @classmethod
    def from_config(cls, config: ABPCHGConfig) -> "ABPCHGConfig":
        """Construct an ``ABPCHGConfig`` instance from an ``ABPCHGConfig`` dataclass.

        Uses ``getattr`` with dataclass defaults for every field so that
        partially specified configs remain valid — fields absent from the
        config object silently fall back to the defaults defined above.

        Args:
            config: ``ABPCHGConfig`` dataclass populated from YAML, CLI flags,
                or the experiment registry.  Only fields that exist on the
                config object are read; missing fields use class defaults.

        Returns:
            Fully resolved ``ABPCHGConfig`` instance.
        """
        return cls(
            gamma=getattr(config, "gamma", 0.95),
            seed=getattr(config, "seed", None),
            overflow_penalty=getattr(config, "overflow_penalty", 2.0),
            ph_base_rho=getattr(config, "ph_base_rho", 1.0),
            ph_max_iterations=getattr(config, "ph_max_iterations", 100),
            ph_convergence_tol=getattr(config, "ph_convergence_tol", 1e-4),
            alns_iterations=getattr(config, "alns_iterations", 50),
            alns_max_routes=getattr(config, "alns_max_routes", 5),
            alns_rc_tolerance=getattr(config, "alns_rc_tolerance", 1e-4),
            alns_remove_fraction=getattr(config, "alns_remove_fraction", 0.25),
            dive_penalty_M=getattr(config, "dive_penalty_M", 10_000.0),
            fo_tabu_length=getattr(config, "fo_tabu_length", 10),
            fo_max_unfix=getattr(config, "fo_max_unfix", 5),
            fo_strategy=getattr(config, "fo_strategy", "overflow_urgency"),
            fo_max_iterations=getattr(config, "fo_max_iterations", 20),
            ml_reliability_c=getattr(config, "ml_reliability_c", 1.0),
            ml_pseudocost_ema_alpha=getattr(config, "ml_pseudocost_ema_alpha", 0.5),
            sc_consensus_threshold=getattr(config, "sc_consensus_threshold", 0.95),
            benders_max_iterations=getattr(config, "benders_max_iterations", 50),
            benders_convergence_tol=getattr(config, "benders_convergence_tol", 1e-3),
            benders_cut_pool_max=getattr(config, "benders_cut_pool_max", 500),
            max_visits_per_bin=getattr(config, "max_visits_per_bin", 1),
            theta_upper_bound=getattr(config, "theta_upper_bound", 1e6),
            gurobi_master_time_limit=getattr(config, "gurobi_master_time_limit", 60.0),
            gurobi_sub_time_limit=getattr(config, "gurobi_sub_time_limit", 30.0),
            gurobi_mip_gap=getattr(config, "gurobi_mip_gap", 1e-4),
            gurobi_output_flag=getattr(config, "gurobi_output_flag", False),
            subproblem_relax=getattr(config, "subproblem_relax", True),
        )
