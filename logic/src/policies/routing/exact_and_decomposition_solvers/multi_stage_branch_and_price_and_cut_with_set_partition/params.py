"""Configuration parameters for the Multi-Stage Branch-and-Price-and-Cut with Set Partitioning (MSBPCSP) solver.

Based on the work of Barnhart et al. (1998, 2000) and standard exact VRPP protocols.

Attributes:
    MSBPCSPParams (class): Data container for solver settings.

Example:
    >>> params = MSBPCSPParams(time_limit=120.0, optimality_gap=1e-6)
    >>> dict_params = params.to_dict()
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict, Optional


@dataclass
class MSBPCSPParams:
    """Configuration parameters for the Multi-Stage Branch-and-Price-and-Cut with Set Partitioning solver.

    Attributes:
        time_limit (float): Max wall-clock time in seconds.
        profit_aware_operators (bool): Enable VRPP profit logic in subproblems.
        vrpp (bool): Problem type flag.
        seed (Optional[int]): Random seed.
        search_strategy (str): Tree search type (depth_first, etc).
        cutting_planes (str): Cut type (rcc, etc).
        branching_strategy (str): Branching rule (divergence, etc).
        max_cg_iterations (int): Max CG rounds per node.
        max_cut_iterations (int): Max cut rounds per node.
        max_cuts_per_iteration (int): Max cuts added per round.
        max_routes_per_pricing (int): Max columns added per pricing call.
        max_bb_nodes (int): Global node limit.
        optimality_gap (float): Relative gap for proven optimality.
        early_termination_gap (float): Gap at which we stop the search.
        use_ng_routes (bool): Whether to use ng-route relaxation in pricing.
        ng_neighborhood_size (int): Size of ng-neighborhoods for relaxation.
        enable_heuristic_rcc_separation (bool): Whether to enable fractional RCC separation.
        enable_comb_cuts (bool): Whether to enable heuristic comb inequalities.
        cut_orthogonality_threshold (float): Cosine similarity ceiling for cut filtering.
        use_spatial_partitioning (bool): Enable spatial domain partitioning.
        enable_strong_branching_heuristic (bool): Use fast divergence-based branching.
        enable_column_pool_deduplication (bool): Whether to enable column pool deduplication.
        rc_tolerance (float): Minimum reduced cost to accept a new column.
        exact_mode (bool): Whether to enable strict exact management.
        strong_branching_size (int): Number of candidates to evaluate in strong branching.
        cg_at_root_only (bool): Generate columns only at the root node.
        rcspp_timeout (float): Max time for a single pricer call.
        rcspp_max_labels (int): Max labels in RCSPP solver.
        lr_pre_pruning (bool): Enable Lagrangian pruning.
        lr_lambda_init (float): Initial multiplier.
        lr_max_subgradient_iters (int): Max iterations for LR.
        lr_subgradient_theta (float): Step size multiplier.
        lr_op_time_limit (float): Time limit for OP solver in LR.
        lr_pre_pruning_depth_limit (int): Max depth for LR.
        lr_warm_start_cg (bool): Seed CG with LR result.
    """

    time_limit: float = 60.0
    profit_aware_operators: bool = False
    vrpp: bool = True
    seed: Optional[int] = None
    search_strategy: str = "depth_first"
    # Bug #1 fix: paper §4 uses Lifted Cover Inequalities on saturated arcs as its
    # sole cut family. "rcc" (Rounded Capacity Cuts, Lysgaard et al. 2004) post-dates
    # the paper and is a completely different family.  "saturated_arc_lci" matches §6
    # exactly and replicates the Tables 3–4 experimental setup.
    # Set to "rcc" (or "all") for VRPP-specific tightening beyond the paper.
    cutting_planes: str = "saturated_arc_lci"
    branching_strategy: str = "divergence"
    max_cg_iterations: int = 50
    max_cut_iterations: int = 5
    max_cuts_per_iteration: int = 5
    max_routes_per_pricing: int = 5
    max_bb_nodes: int = 1000
    optimality_gap: float = 1e-4
    early_termination_gap: float = 1e-3  # HEURISTIC SHORTCUT: See docstring for caveats.
    use_ng_routes: bool = True
    ng_neighborhood_size: int = 8
    enable_heuristic_rcc_separation: bool = True
    enable_comb_cuts: bool = False
    cut_orthogonality_threshold: float = 0.8
    use_spatial_partitioning: bool = False
    # Disabled by default. The current implementation is a heuristic proxy
    # that returns the top pre-sorted divergence candidate without solving
    # any child LP relaxations. When True, it bypasses fleet-size and
    # Ryan-Foster branching levels at every fractional node for no benefit.
    # Enable only after implementing full LP-based strong branching
    # (2×N child LP solves per candidate).
    enable_strong_branching_heuristic: bool = False
    enable_column_pool_deduplication: bool = True
    rc_tolerance: float = 1e-8  # Fix 8: Minimum reduced cost to accept a new column
    exact_mode: bool = False  # Task 3: Enable strict exact management
    strong_branching_size: int = 5  # Task 1: Number of candidates for strong branching
    cg_at_root_only: bool = False  # Paper Table 2: Run column generation only at root node
    rcspp_timeout: float = 30.0  # Safety cap for single pricer call
    rcspp_max_labels: int = 1000000  # Safety cap to prevent OOM

    # Paper §3.2 (Node Selection) explicitly states the DFS always explores
    # the side where the shorter path p is still allowed — i.e., the child that forbids
    # A(d, a2) (the arc set of the longer path). This takes priority over LP-bound hints.
    prefer_shorter_path_dfs: bool = True

    # ---------------------------------------------------------------------------
    # Lagrangian Relaxation Pre-Pruning
    # ---------------------------------------------------------------------------
    # When enabled, a fast subgradient pass is run at each B&B node before column
    # generation. If the Lagrangian bound is dominated by the incumbent, the node
    # is pruned without touching the master LP.
    #
    # Theoretical basis: L(λ*) ≥ z* (VRPP optimum) for all λ ≥ 0. If the CG LP
    # bound is the true Lagrangian dual (as proven by Dantzig-Wolfe equivalence),
    # then for tightly converged nodes the LR bound adds nothing. However at early
    # B&B nodes — especially in deep subtrees with heavy branching fixings — a
    # fast LR pass over the reduced customer set can prune before CG starts.
    # ---------------------------------------------------------------------------
    lr_pre_pruning: bool = False
    """Enable Lagrangian pre-pruning at B&B nodes before column generation."""

    lr_lambda_init: float = 0.0
    """Initial Lagrange multiplier λ₀ ≥ 0 for the capacity constraint."""

    lr_max_subgradient_iters: int = 30
    """Max Polyak iterations per node. Keep low (20–40) since this runs at every node."""

    lr_subgradient_theta: float = 1.0
    """Step-size multiplier θ ∈ (0, 2] for the Polyak rule."""

    lr_op_time_limit: float = 3.0
    """Per-solve wall-clock budget (seconds) for the uncapacitated OP inner solver."""

    lr_pre_pruning_depth_limit: int = -1
    """Maximum B&B tree depth at which to apply LR pruning.
    -1 means apply at every depth. Set to 0 to restrict to root only.
    Useful because LR bounds weaken at deep nodes due to many forced-in/out fixings."""

    lr_warm_start_cg: bool = False
    """If True, use λ* from subgradient to seed initial columns before CG starts.
    Calls solve_uncapacitated_op at λ* and adds the resulting route to the column
    pool. Provides a stronger starting point for CG at the cost of one extra OP solve.
    Only meaningful when lr_pre_pruning is also True (shares the λ* already computed)."""

    @classmethod
    def from_config(cls, config: Any) -> MSBPCSPParams:
        """Create BPCParams from a configuration object or dictionary.

        Performs explicit type casting for numeric fields to ensure compatibility
        with Hydra/YAML scientific notation which might be loaded as strings.

        Args:
            config: Configuration source (dict or Hydra config).

        Returns:
            MSBPCSPParams: Initialized parameter object.
        """
        if config is None:
            return cls()

        raw_data: Dict[str, Any] = {}
        if isinstance(config, dict):
            raw_data = config
        else:
            # Handle Hydra DictConfig or other object types
            for f in fields(cls):
                if hasattr(config, f.name):
                    raw_data[f.name] = getattr(config, f.name)

        kwargs: Dict[str, Any] = {}
        for f in fields(cls):
            val = raw_data.get(f.name, f.default)
            if val is MISSING:
                if f.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                    val = f.default_factory()  # type: ignore[misc]
                else:
                    continue

            # Explicit type casting
            if val is not None:
                if f.type is float or f.type == "float":
                    val = float(val)
                elif f.type is int or f.type == "int":
                    val = int(val)
                elif f.type is bool or f.type == "bool":
                    val = val.lower() in ("true", "1", "yes") if isinstance(val, str) else bool(val)

            kwargs[f.name] = val

        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert BPCParams to a dictionary for backend compatibility.

        Returns:
            Dict[str, Any]: Mapping of parameter names to values.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
