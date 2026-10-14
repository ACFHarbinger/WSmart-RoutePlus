"""
Configuration parameters for ABPC-HG.

Attributes:
    ABPCHGParams: configuration for Adaptive Branch-and-Price-and-Cut with Heuristic Guidance.

Example:
    >>> from logic.src.policies.routing.adaptive_branch_and_price_and_cut_with_heuristic_guidance.params import ABPCHGParams
    >>> params = ABPCHGParams()
    >>> print(params)
    ABPCHGParams(gamma=0.95, seed=None, overflow_penalty=2.0, ph_base_rho=1.0, ph_max_iterations=100,
        ph_convergence_tol=0.0001, alns_iterations=50, alns_max_routes=5, alns_rc_tolerance=0.0001,
        alns_remove_fraction=0.25, dive_penalty_M=10000.0, ive_root_time_limit=600.0, dive_min_subproblems=1,
        dive_max_subproblems=25, dive_min_per_subproblem=0, dive_max_per_subproblem=25, fo_root_time_limit=600.0,
        fo_max_improvement=0.001, fo_warmup_routes=50, fo_batch_size=10, ml_branching_pool_size=128,
        ml_node_pool_size=1000, ml_target_depth=32, ml_min_reliable_decisions=4, ml_reliability_threshold=0.55,
        ml_reliability_decay=0.5, ml_eval_batch_size=128, ml_train_epochs=10, ml_train_patience=5,
        sc_branch_fraction=0.7, sc_depth_threshold=128, sc_reliability_decay=0.3, sc_reliability_threshold=0.75,
        sc_min_reliable_decisions=4, benders_max_iterations=150, benders_lower_bound_improvement_tol=0.0005,
        benders_upper_bound_improvement_tol=0.0002, benders_no_improvement_window=8, benders_early_stop_fraction=0.6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from logic.src.configs.policies.abpc_hg import ABPCHGConfig


@dataclass
class ABPCHGParams:
    r"""
    Comprehensive configuration for Adaptive Branch-and-Price-and-Cut with
    Heuristic Guidance (ABPC-HG).

    Parameters are grouped by the pipeline component that consumes them.
    All defaults are calibrated for practical performance on medium-scale
    stochastic MPVRPP instances (50–200 bins, 5–10 day horizons).

    Component Map
    -------------
    ┌─────────────────────────────────┬───────────────────────────────────┐
    │ Component                       │ Parameter prefix / group          │
    ├─────────────────────────────────┼───────────────────────────────────┤
    │ ScenarioPrizeEngine             │ gamma, overflow_penalty           │
    │ ProgressiveHedgingCGLoop        │ ph_*                              │
    │ ALNSMultiPeriodPricer           │ alns_*                            │
    │ DiveAndPricePrimalHeuristic     │ dive_*                            │
    │ FixAndOptimizeRefiner           │ fo_*                              │
    │ MLBranchingStrategy             │ ml_*                              │
    │ ScenarioConsistentBranching     │ sc_*                              │
    │ TemporalBendersCoordinator      │ benders_*                         │
    └─────────────────────────────────┴───────────────────────────────────┘

    Notes on sign conventions
    -------------------------
    This codebase follows the VRPP maximisation convention throughout:
    a column with *positive* reduced cost is improving.  Where a comment
    references the minimisation dual, the sign is flipped accordingly.

    Attributes:
        gamma (float): Inter-day discount factor γ ∈ (0, 1] applied to future
            value estimates in ScenarioPrizeEngine and the Bellman recursion.

        seed (Optional[int]): Global RNG seed.

        overflow_penalty (float): Revenue multiplier applied when a bin is
            projected to overflow before the next feasible visit (i.e.
            days_to_overflow ≤ 1).  The effective penalty is:
            −overflow_penalty × τ × revenue.

        ph_base_rho (float): Base augmented-Lagrangian penalty coefficient ρ₀ for the PH proximal
            term added to each per-scenario reduced cost: c'_{k,ξ} += −ρ_eff · (x_k − x̄_k)²

        ph_max_iterations (int): Maximum number of PH consensus iterations before declaring convergence
            of the root-node multi-scenario CG.  Each iteration involves one solve
            per scenario RMP followed by an x̄ update.

        ph_convergence_tol (float): Convergence tolerance for the PH loop.  Iteration stops when the
            normalised consensus residual satisfies:

                max_ξ  ‖x^ξ − x̄‖ / max(1, ‖x̄‖)  <  tol

            Tighter tolerances improve the quality of the non-anticipative LP bound
            but increase root-node solve time.

        alns_iterations (int): Maximum number of ALNS outer-loop iterations (per B&P node).

        alns_max_routes (int): Target number of routes to decompose into when building

        alns_rc_tolerance (float): Tolerance on the duality gap (dual - LP value).  The ALNS loop
            terminates when the gap closes below this fraction of the current LP bound:

                (UB_ALNS - LB_LP) / max(1, LB_LP)  <  tol

            Low tolerances force tighter integration between ALNS and PH but increase the number
            of ALNS iterations required to close the gap.

        alns_remove_fraction (float): Fraction of customer-arcs to remove in each destroy step,
            drawn uniformly from the current ALNS solution.

        dive_penalty_M (float): Large constant M used in the penalty formulation for route
            feasibility violations (vehicle capacity, time windows) within the DiveAndPricePrimalHeuristic.

        fo_tabu_length (int): Length of the tabu list used in the FixAndOptimizeRefiner.

        fo_max_unfix (int): Maximum number of arcs to unfix in each iteration of the FixAndOptimizeRefiner.

        fo_strategy (str): Strategy used to select arcs to fix and unfix in the FixAndOptimizeRefiner.
            Possible values are: 'ml', 'greedy', 'random'.

        fo_max_iterations (int): Maximum number of iterations of the FixAndOptimizeRefiner.

        ml_reliability_c (float): Blending coefficient c in the reliability branching score function.

        ml_pseudocost_ema_alpha (float): Exponential moving-average coefficient α for pseudo-cost updates after
            each exact strong-branching evaluation: Ψ_new = α · Ψ_old + (1 − α) · Δ_observed
            α = 0.5 gives equal weight to history and new observations.  Lower α
            makes pseudo-costs more adaptive to recent LP changes; higher α
            stabilises estimates in the presence of noisy LP solutions.

        sc_consensus_threshold (float): Base consensus threshold above which a fractional
            variable is considered 'strongly agreed upon' across deterministic scenario solutions.

        benders_max_iterations (int): Maximum number of Benders master–subproblem iteration cycles before
            terminating and returning the best primal solution found.

        benders_convergence_tol (float): Primal–dual gap tolerance for early Benders termination:
            (UB_current - LB_LP) / max(1, UB_current) < tol

        benders_cut_pool_max (int): Maximum number of Benders cuts to store in the cut pool.

        max_visits_per_bin (int): Maximum number of visits per bin.

        theta_upper_bound (float): Upper bound on θ used in the node prize calculation.

        gurobi_master_time_limit (float): Lower bound on θ used in the node prize calculation.

        gurobi_sub_time_limit (float): Number of subproblems in the Branch-and-Price-and-Cut algorithm.

        gurobi_mip_gap (float): Tolerance for the MIP gap.  The algorithm terminates when the
            relative gap between the upper bound (best integer solution) and the
            lower bound (relaxed master LP) is smaller than this value.

        gurobi_output_flag (bool): Whether to output Gurobi solver logs to the console.

        subproblem_relax (bool): Whether to solve the subproblems as LPs (True) or MIPs (False).
    """

    gamma: float = 0.95
    """Inter-day discount factor γ ∈ (0, 1] applied to future value estimates
    in ScenarioPrizeEngine and the Bellman recursion.  Values closer to 1.0
    weight distant-day revenue almost equally to today's; lower values
    concentrate collection effort on the near horizon."""

    seed: Optional[int] = None
    """Global RNG seed propagated to all stochastic components (ALNS destroy
    operator, scenario sampling).  ``None`` yields non-deterministic runs."""

    # ------------------------------------------------------------------ #
    # ScenarioPrizeEngine                                                  #
    # ------------------------------------------------------------------ #

    overflow_penalty: float = 2.0
    """Revenue multiplier applied when a bin is projected to overflow before
    the next feasible visit (i.e. days_to_overflow ≤ 1).  The effective
    penalty is: −overflow_penalty × τ × revenue.  Raising this sharpens
    the urgency signal embedded in node prizes, driving earlier service of
    high-fill bins at the cost of potentially suboptimal routing."""

    # ------------------------------------------------------------------ #
    # Progressive Hedging — root-node CG non-anticipativity               #
    # ------------------------------------------------------------------ #

    ph_base_rho: float = 1.0
    """Base augmented-Lagrangian penalty coefficient ρ₀ for the PH proximal
    term added to each per-scenario reduced cost:

        c'_{k,ξ} += −ρ_eff · (x_k − x̄_k)²

    The effective ρ per variable is scaled dynamically by the cross-scenario
    prize variance: ρ_eff = ρ₀ · (1 + Var_ξ[π_i^ξ]).  Higher values enforce
    stronger non-anticipativity at the cost of slower convergence in the
    per-scenario RMPs."""

    ph_max_iterations: int = 100
    """Maximum number of PH consensus iterations before declaring convergence
    of the root-node multi-scenario CG.  Each iteration involves one solve
    per scenario RMP followed by an x̄ update."""

    ph_convergence_tol: float = 1e-4
    """Convergence tolerance for the PH loop.  Iteration stops when the
    normalised consensus residual satisfies:

        max_ξ  ‖x^ξ − x̄‖ / max(1, ‖x̄‖)  <  tol

    Tighter tolerances improve the quality of the non-anticipative LP bound
    but increase root-node solve time."""

    # ------------------------------------------------------------------ #
    # ALNS Pricing                                                         #
    # ------------------------------------------------------------------ #

    alns_iterations: int = 50
    """Number of destroy-and-repair cycles executed per initial route seed
    in each ALNS pricing call.  More iterations improve column quality at
    the cost of pricing time; the exact ESPPRC fallback compensates when
    ALNS is insufficient."""

    alns_max_routes: int = 5
    """Maximum number of improving columns returned per ALNS pricing call.
    Once this many routes with reduced cost > ``alns_rc_tolerance`` are
    found, further ALNS iterations for the current seed are skipped."""

    alns_rc_tolerance: float = 1e-4
    """Reduced-cost threshold (VRPP maximisation: rc > tol is improving).
    Columns with rc ≤ tol are discarded to avoid near-degenerate pivots.
    Corresponds to the standard −1e-4 threshold in the minimisation dual."""

    alns_remove_fraction: float = 0.25
    """Fraction of non-depot nodes removed per destroy step:
    num_remove = max(1, ⌊|route| × fraction⌋).  Larger fractions yield
    more diversified exploration; smaller fractions focus on local refinement."""

    # ------------------------------------------------------------------ #
    # Dive-and-Price primal heuristic                                      #
    # ------------------------------------------------------------------ #

    dive_penalty_M: float = 10_000.0
    """Big-M soft penalty applied to a column that induces infeasibility
    during the diving procedure.  The penalised column is retained in the
    RMP but its objective coefficient is reduced by M, effectively
    discouraging its selection while preserving feasibility without
    backtracking."""

    # ------------------------------------------------------------------ #
    # Fix-and-Optimize corridor method                                     #
    # ------------------------------------------------------------------ #

    fo_tabu_length: int = 10
    """Number of recently unfixed clusters retained in the tabu list.
    A candidate cluster is rejected if its Jaccard similarity with any
    tabu entry exceeds 0.8, preventing short-cycle revisits during the
    Fix-and-Optimize refinement phase."""

    fo_max_unfix: int = 5
    """Maximum number of bins simultaneously released in each corridor.
    Controls the size of the sub-MIP handed to the exact BPC engine: larger
    values yield better improvement per iteration at the cost of solve time."""

    fo_strategy: Literal["overflow_urgency", "scenario_divergence"] = "overflow_urgency"
    """Cluster selection strategy for Fix-and-Optimize:

    ``overflow_urgency``
        Ranks bins by ascending days-to-overflow.  Prioritises bins most
        at risk of unserved overflow, improving solution feasibility under
        capacity constraints.

    ``scenario_divergence``
        Ranks bins by descending cross-scenario fill variance.  Focuses
        refinement on bins with the highest uncertainty, improving expected
        profit robustness across the scenario tree."""

    fo_max_iterations: int = 20
    """Maximum number of Fix-and-Optimize corridor passes over the incumbent
    solution per full horizon solve.  The loop exits early if the tabu list
    exhausts all distinct clusters of size ``fo_max_unfix``."""

    # ------------------------------------------------------------------ #
    # ML / Reliability branching                                           #
    # ------------------------------------------------------------------ #

    ml_reliability_c: float = 1.0
    """Blending coefficient c in the reliability branching score function:

        Score(xᵢ) = min(Ψ↓ᵢ, Ψ↑ᵢ) + c · max(Ψ↓ᵢ, Ψ↑ᵢ)

    where Ψ↓ᵢ, Ψ↑ᵢ are historical pseudo-costs for branching down/up on
    variable i.  c = 1 gives equal weight to both directions (product score
    variant); c > 1 penalises high asymmetry."""

    ml_pseudocost_ema_alpha: float = 0.5
    """Exponential moving-average coefficient α for pseudo-cost updates after
    each exact strong-branching evaluation:

        Ψ_new = α · Ψ_old + (1 − α) · Δ_observed

    α = 0.5 gives equal weight to history and new observations.  Lower α
    makes pseudo-costs more adaptive to recent LP changes; higher α
    stabilises estimates in the presence of noisy LP solutions."""

    # ------------------------------------------------------------------ #
    # Scenario-Consistent Branching                                        #
    # ------------------------------------------------------------------ #

    sc_consensus_threshold: float = 0.95
    """Base consensus threshold above which a fractional variable is
    considered 'strongly agreed upon' across deterministic scenario solutions.
    Used as a reference for horizon-adaptive threshold decay: early in the
    horizon (many days remaining) the effective threshold is close to this
    value; later it relaxes, accepting weaker consensus as the problem
    becomes more deterministic."""

    # ------------------------------------------------------------------ #
    # Temporal Benders coordination                                        #
    # ------------------------------------------------------------------ #

    benders_max_iterations: int = 50
    """Maximum number of Benders master–subproblem iteration cycles before
    terminating and returning the best primal solution found.  Each iteration
    solves the master MIP, then one subproblem per (day, scenario) pair, and
    adds probability-weighted optimality cuts."""

    benders_convergence_tol: float = 1e-3
    """Primal–dual gap tolerance for early Benders termination:

        (UB − LB) / max(1, |UB|) < tol

    Tighter tolerances approach the true optimal but require more iterations;
    looser tolerances trade off solution quality for speed."""

    benders_cut_pool_max: int = 500
    """Maximum number of Benders optimality cuts retained in the master
    problem cut pool.  When the pool exceeds this limit, the oldest cuts
    are purged (FIFO).  Larger pools improve the master LP bound but
    increase per-iteration master solve time."""

    # ------------------------------------------------------------------ #
    # Gurobi solver settings                                               #
    # ------------------------------------------------------------------ #

    max_visits_per_bin: int = 1
    """Maximum number of times a single bin may be assigned across the entire
    planning horizon: Σ_d z[i,d] ≤ max_visits_per_bin.  The default of 1
    corresponds to the standard once-per-horizon MPVRPP variant.  Increase
    for problems where high-fill bins may need service multiple times."""

    theta_upper_bound: float = 1e6
    """Initial upper bound imposed on each surrogate variable θ[d] in the
    master MIP.  Prevents the master from being unbounded before the first
    Benders cuts are added.  A safe choice is the sum of all node prizes for
    the highest-prize day; the default of 1e6 is suitable for most
    waste-collection instances with revenue in the range [0, 1e4]."""

    gurobi_master_time_limit: float = 60.0
    """Maximum Gurobi solve time (seconds) per master MIP call.  The master is
    re-solved once per Benders iteration; tighter limits trade solution quality
    for speed.  If the time limit is hit with a feasible solution, the best
    incumbent is used."""

    gurobi_sub_time_limit: float = 30.0
    """Maximum Gurobi solve time (seconds) per subproblem LP (or MIP) call.
    Subproblems are solved once per (day, scenario) pair per Benders iteration,
    so the total time scales as T × |Ξ| × benders_max_iterations."""

    gurobi_mip_gap: float = 1e-4
    """Relative MIP optimality gap for Gurobi master and subproblem (when MIP
    mode is used).  Tighter values improve Benders bound quality at the cost
    of longer solve times."""

    gurobi_output_flag: bool = False
    """Enable Gurobi console output for master and subproblem models.  Set to
    ``True`` for debugging; suppressed by default to keep logs clean."""

    subproblem_relax: bool = True
    """Subproblem solving mode:

    ``True`` (default): Solve the LP relaxation (no subtour elimination).
        Faster; gives valid but potentially loose Benders cuts.

    ``False``: Solve a MIP with MTZ subtour elimination, then fix the integer
        solution and re-solve as an LP for exact dual recovery.  Slower but
        produces tighter cuts, accelerating Benders convergence at the cost
        of higher per-iteration time."""

    @classmethod
    def from_config(cls, config: ABPCHGConfig) -> ABPCHGParams:
        """Construct an ``ABPCHGParams`` instance from an ``ABPCHGConfig`` dataclass.

        Uses ``getattr`` with dataclass defaults for every field so that
        partially specified configs remain valid — fields absent from the
        config object silently fall back to the defaults defined above.

        Args:
            config: ``ABPCHGConfig`` dataclass populated from YAML, CLI flags,
                or the experiment registry.  Only fields that exist on the
                config object are read; missing fields use class defaults.

        Returns:
            Fully resolved ``ABPCHGParams`` instance.
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
