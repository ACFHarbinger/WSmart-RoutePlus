"""
BPC (Branch-and-Price-and-Cut) configuration.

Attributes:
    BPCConfig: Attributes for BPC configuration.

Example:
    >>> from configs.policies.bpc import BPCConfig
    >>> config = BPCConfig()
    >>> config.time_limit
    60.0
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.mandatory_selection import MandatorySelectionConfig
from .other.route_improvement import RouteImprovingConfig


@dataclass
class BPCConfig:
    """Configuration for Branch-and-Price-and-Cut (BPC) policy.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        profit_aware_operators: Whether to use profit-aware operators.
        vrpp: Whether to use VRPP.
        seed: Random seed for reproducibility.
        mandatory_selection: List of mandatory strategy config files.
        route_improvement: List of route improvement operations to apply.
        search_strategy: B&B node selection strategy ('best_first' or 'depth_first').
        cutting_planes: Cutting plane family ('rcc', 'sri', 'lci', or 'all').
        branching_strategy: Branching rule ('ryan_foster', 'edge', or 'divergence').
        max_cg_iterations: Maximum iterations for column generation loop.
        max_cuts_per_iteration: Maximum cuts to add per iteration.
        max_routes_per_pricing: Maximum routes to add per pricing call.
        max_bb_nodes: Maximum nodes to explore in the B&B tree.
        optimality_gap: Relative gap for proven optimality.
        early_termination_gap: Gap at which to stop search early (heuristic shortcut).
        use_ng_routes: Whether to use ng-route relaxation in pricing.
        ng_neighborhood_size: Size of ng-neighborhoods for relaxation.
        enable_heuristic_rcc_separation: Whether to enable (heuristic) fractional RCC separation.
        enable_comb_cuts: Whether to enable heuristic comb inequalities.
        cut_orthogonality_threshold: Cosine similarity ceiling for cut filtering.
        enable_strong_branching_heuristic: Whether to enable heuristic strong branching.
        enable_column_pool_deduplication: Whether to enable column pool deduplication.
        rc_tolerance: Minimum reduced cost to accept a new column.
        exact_mode: Whether to enable strict exact management.
        strong_branching_size: Number of candidates for strong branching.
        cg_at_root_only: Run column generation only at the root B&B node; at all
            descendant nodes, no new columns are generated (use the pool built at
            root).  Replicates the "CG at root only" experiment in Table 2 of
            Barnhart, Hane, and Vance (2000).  Trades solution quality for speed.
        use_swc_tcf_heuristic_pricing: Whether to use SWC-TCF as a fast heuristic pricer.
        use_swc_tcf_primal_heuristic: Whether to use SWC-TCF as a primal heuristic at nodes.
        rcspp_timeout: Safety cap for single pricer call (seconds).
        rcspp_max_labels: Safety cap to prevent OOM in RCSPP.
        lr_pre_pruning: Whether to enable pre-pruning using the Lagrangian Relaxation (LR) heuristic.
        lr_lambda_init: Initial lambda value for LR heuristic.
        lr_max_subgradient_iters: Maximum iterations for LR subgradient updates.
        lr_subgradient_theta: Step size parameter for LR subgradient updates.
        lr_op_time_limit: Time limit for LR operations (seconds).
        lr_pre_pruning_depth_limit: Depth limit for LR pre-pruning.
        lr_warm_start_cg: Whether to warm-start column generation with LR solutions.
        prefer_shorter_path_dfs: Whether to prefer the child where the shorter path is still allowed.
        enable_node_visitation_branching: Whether to enable hierarchical node-visitation (y_v) branching before arc/RF branching.
        enable_dssr: Whether to enable Decremental State-Space Relaxation in pricing (Righini, Salani 2008).
        dssr_max_iters: Maximum DSSR refinement iterations per pricing call.
        enable_reduced_cost_arc_fixing: Whether to enable reduced-cost arc elimination from pricing graph (Irnich et al. 2010).
        route_budget: Maximum route cost/distance budget for conflict-cut and path-elimination separation.
    """

    time_limit: float = 60.0
    profit_aware_operators: bool = False
    vrpp: bool = True
    seed: Optional[int] = None
    mandatory_selection: Optional[List[MandatorySelectionConfig]] = None
    route_improvement: Optional[List[RouteImprovingConfig]] = None
    search_strategy: str = "depth_first"
    cutting_planes: str = "rcc"
    branching_strategy: str = "divergence"
    max_cg_iterations: int = 50
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
    enable_strong_branching_heuristic: bool = False  # see policy_bpc.yaml
    enable_column_pool_deduplication: bool = True
    rc_tolerance: float = 1e-5
    exact_mode: bool = False
    strong_branching_size: int = 5
    cg_at_root_only: bool = False
    lr_pre_pruning: bool = False
    lr_lambda_init: float = 0.0
    lr_max_subgradient_iters: int = 30
    lr_subgradient_theta: float = 1.0
    lr_op_time_limit: float = 3.0
    lr_pre_pruning_depth_limit: int = -1
    lr_warm_start_cg: bool = False
    rcspp_timeout: float = 30.0  # Safety cap for single pricer call
    rcspp_max_labels: int = 1000000  # Safety cap to prevent OOM
    prefer_shorter_path_dfs: bool = True
    # ── Tier 1 + Selection-Routing Duality Parameters ──────────────────────
    enable_node_visitation_branching: bool = True
    """Enable hierarchical node-visitation (y_v) branching before arc/RF branching.
    Directly resolves the VRPP profit-selection × routing duality by branching
    on which nodes to visit (Boussier, Feillet, Gendreau 2007; Pessoa et al. 2020)."""

    enable_dssr: bool = True
    """Enable Decremental State-Space Relaxation in pricing (Righini, Salani 2008).
    Iteratively adds cycled vertices to ng-memory until returned paths are elementary."""

    dssr_max_iters: int = 8
    """Maximum DSSR refinement iterations per pricing call."""

    enable_reduced_cost_arc_fixing: bool = True
    """Enable reduced-cost arc elimination from pricing graph (Irnich et al. 2010).
    Arcs whose min reduced cost exceeds the LP-incumbent gap are pruned."""

    route_budget: float = float("inf")
    """Maximum route cost/distance budget for conflict-cut and path-elimination separation.
    Set to vehicle travel-time limit if applicable."""
