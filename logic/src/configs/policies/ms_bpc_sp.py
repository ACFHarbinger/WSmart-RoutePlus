"""
Multi-Stage Branch-and-Price-and-Cut with Set Partitioning (MSBPCSP) configuration.

Attributes:
    MSBPCSPConfig: Attributes for MSBPCSP configuration.

Example:
    >>> from configs.policies.ms_bpc_sp import MSBPCSPConfig
    >>> config = MSBPCSPConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.packing import PackingConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class MSBPCSPConfig:
    """Configuration for Multi-Stage Branch-and-Price-and-Cut with Set Partitioning policy.

    Attributes:
        time_limit: Maximum wall-clock time in seconds for the solver.
        profit_aware_operators: Enable VRPP profit logic in subproblems.
        seed: Random seed for reproducibility.
        packing: List of packing algorithm config files.
        route_improvement: List of route improvement operations to apply.
        search_strategy: Tree search type ('best_first', 'depth_first', or 'hybrid').
        cutting_planes: Cut type ('saturated_arc_lci', 'rcc', 'sri', or 'all').
        branching_strategy: Branching rule ('ryan_foster', 'edge', or 'divergence').
        max_cg_iterations: Maximum iterations for column generation loop.
        max_cut_iterations: Maximum iterations for cutting plane loop per CG iteration.
        max_cuts_per_iteration: Maximum cuts to add per iteration.
        max_routes_per_pricing: Maximum routes to add per pricing call.
        max_bb_nodes: Maximum nodes to explore in the B&B tree.
        optimality_gap: Relative gap for proven optimality.
        early_termination_gap: Gap at which to stop search early (heuristic shortcut).
        use_ng_routes: Whether to use ng-route relaxation in pricing.
        ng_neighborhood_size: Size of ng-neighborhoods for relaxation.
        enable_heuristic_rcc_separation: Whether to enable fractional RCC separation.
        enable_comb_cuts: Whether to enable heuristic comb inequalities.
        cut_orthogonality_threshold: Cosine similarity ceiling for cut filtering.
        use_spatial_partitioning: Whether to use spatial partitioning for branching.
        enable_strong_branching_heuristic: Whether to enable heuristic strong branching.
        enable_column_pool_deduplication: Whether to enable column pool deduplication.
        rc_tolerance: Minimum reduced cost to accept a new column.
        exact_mode: Whether to enable strict exact management.
        strong_branching_size: Number of candidates for strong branching.
        cg_at_root_only: Run column generation only at the root B&B node.
        rcspp_timeout: Safety cap for single pricer call (seconds).
        rcspp_max_labels: Safety cap to prevent OOM in RCSPP.
        prefer_shorter_path_dfs: Whether to prefer the child where the shorter path is still allowed.
        lr_pre_pruning: Enable Lagrangian pre-pruning at B&B nodes.
        lr_lambda_init: Initial Lagrange multiplier λ₀ ≥ 0.
        lr_max_subgradient_iters: Maximum iterations for LR subgradient updates.
        lr_subgradient_theta: Step size multiplier θ ∈ (0, 2].
        lr_op_time_limit: Time limit for LR operations (seconds).
        lr_pre_pruning_depth_limit: Depth limit for LR pre-pruning.
        lr_warm_start_cg: Whether to warm-start column generation with LR solutions.
    """

    time_limit: float = 60.0
    profit_aware_operators: bool = False
    seed: Optional[int] = None
    packing: Optional[List[PackingConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    search_strategy: str = "depth_first"
    cutting_planes: str = "saturated_arc_lci"
    branching_strategy: str = "divergence"
    max_cg_iterations: int = 50
    max_cut_iterations: int = 5
    max_cuts_per_iteration: int = 5
    max_routes_per_pricing: int = 5
    max_bb_nodes: int = 1000
    optimality_gap: float = 1e-4
    early_termination_gap: float = 1e-3
    use_ng_routes: bool = True
    ng_neighborhood_size: int = 8
    enable_heuristic_rcc_separation: bool = True
    enable_comb_cuts: bool = False
    cut_orthogonality_threshold: float = 0.8
    use_spatial_partitioning: bool = False
    enable_strong_branching_heuristic: bool = False
    enable_column_pool_deduplication: bool = True
    rc_tolerance: float = 1e-8
    exact_mode: bool = False
    strong_branching_size: int = 5
    cg_at_root_only: bool = False
    rcspp_timeout: float = 30.0
    rcspp_max_labels: int = 1000000
    prefer_shorter_path_dfs: bool = True
    lr_pre_pruning: bool = False
    lr_lambda_init: float = 0.0
    lr_max_subgradient_iters: int = 30
    lr_subgradient_theta: float = 1.0
    lr_op_time_limit: float = 3.0
    lr_pre_pruning_depth_limit: int = -1
    lr_warm_start_cg: bool = False
