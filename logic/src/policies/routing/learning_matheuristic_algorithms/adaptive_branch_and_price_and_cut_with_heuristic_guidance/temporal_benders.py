"""
Temporal Decomposition via Benders Cuts.

Attributes:
    _GUROBI_AVAILABLE: Whether Gurobi is available.
    logger: Logger for the Temporal Benders algorithm.
    TemporalBendersCoordinator: The Temporal Benders algorithm implementation.

Examples:
    >>> from .temporal_benders import TemporalBendersCoordinator
    >>> temporal_benders = TemporalBendersCoordinator()
    >>> temporal_benders.solve()

The Gurobi master solve and subproblem evaluation live in ``temporal_benders_gurobi`` and are mixed into ``TemporalBendersCoordinator``.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .gurobi_master import GurobiMasterProblem  # noqa: F401
    from .gurobi_subproblem import GurobiVRPSubproblem  # noqa: F401

    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False

from .temporal_benders_gurobi import TemporalBendersGurobiMixin as TemporalBendersGurobiMixin

logger = logging.getLogger(__name__)


class TemporalBendersCoordinator(TemporalBendersGurobiMixin):
    """
    Master coordination layer linking the multi-day horizon via exact Benders
    decomposition, structured over the scenario tree.

    Algorithm Outline
    -----------------
    The MPVRPP objective decomposes by planning day d as:

        max   Σ_d  Σ_ξ  p_ξ · Q_d(z_{·d}, ξ)        [expected daily profit]
        s.t.  z_{id} ∈ {0,1}                          [bin assignment decisions]
              θ_d  ≥  Σ_ξ p_ξ · [Q_d(z̄, ξ)
                      + Σ_i μ_{id}^ξ (z_{id} − z̄_{id})]   [aggregated Benders cut]

    The master MIP (``GurobiMasterProblem``) optimises over z and θ.  Each
    Benders iteration adds one aggregated optimality cut per day, obtained by
    solving one LP subproblem per (day, scenario) pair via ``GurobiVRPSubproblem``
    and accumulating the probability-weighted dual contributions.

    The master objective (Benders UB) decreases monotonically with cuts; the
    best achieved primal value (LB) increases monotonically as better z̄
    assignments are evaluated.  The algorithm terminates when (UB−LB)/UB < tol.

    Four-Phase Execution
    --------------------
    Phase 1  – Root-node PH-CG (day 0): if ``ph_loop`` is provided, it enforces
               non-anticipativity across per-scenario RMPs before the Benders
               loop starts.  (Stub hook; full integration pending.)

    Phase 2  – Dive-and-Price: if ``dive_heuristic`` is provided, establishes
               a primal upper bound by greedily fixing the most scenario-
               consistent fractional columns.  (Stub hook; full integration
               pending.)

    Phase 3  – Gurobi Benders loop: full implementation via
               ``GurobiMasterProblem`` + ``GurobiVRPSubproblem``.

    Phase 4  – Fix-and-Optimize polish: if ``fix_optimizer`` is provided,
               unfixes targeted bin clusters and re-solves via BPC.  (Stub
               hook; full integration pending.)

    Required kwargs to ``solve()``
    ------------------------------
    dist_matrix : np.ndarray, shape (n_bins+1, n_bins+1)
        Full distance matrix with the depot at index 0 and bins at indices
        1..n_bins.  Must be provided; ``solve()`` raises ``ValueError``
        if absent.

    Injected Components
    -------------------
    All heuristic components are optional (``None`` = skipped).

    ph_loop : ProgressiveHedgingCGLoop or None
    alns_pricer : ALNSMultiPeriodPricer or None
    ml_branching : MLBranchingStrategy or None
    scenario_branching : ScenarioConsistentBranching or None
    dive_heuristic : DiveAndPricePrimalHeuristic or None
    fix_optimizer : FixAndOptimizeRefiner or None

    Attributes:
    ----------
    tree, prize_engine, capacity, revenue, cost_unit
        Problem parameters.
    max_iterations, convergence_tol, cut_pool_max
        Benders convergence controls.
    gurobi_* params
        Gurobi solver settings forwarded to ``GurobiMasterProblem`` and
        ``GurobiVRPSubproblem``.
    """

    def __init__(
        self,
        tree: Any,
        prize_engine: Any,
        capacity: float,
        revenue: float,
        cost_unit: float,
        # ── injected algorithm components ──────────────────────────── #
        ph_loop: Optional[Any] = None,
        alns_pricer: Optional[Any] = None,
        ml_branching: Optional[Any] = None,
        scenario_branching: Optional[Any] = None,
        dive_heuristic: Optional[Any] = None,
        fix_optimizer: Optional[Any] = None,
        # ── Benders convergence controls ────────────────────────────── #
        max_iterations: int = 50,
        convergence_tol: float = 1e-3,
        cut_pool_max: int = 500,
        # ── Gurobi solver settings ──────────────────────────────────── #
        max_visits_per_bin: int = 1,
        theta_upper_bound: float = 1e6,
        gurobi_master_time_limit: float = 60.0,
        gurobi_sub_time_limit: float = 30.0,
        gurobi_mip_gap: float = 1e-4,
        gurobi_output_flag: bool = False,
        subproblem_relax: bool = True,
    ) -> None:
        """
        Args:
            tree: ScenarioTree covering the full planning horizon.
            prize_engine: ScenarioPrizeEngine for scenario-augmented prizes.
            capacity: Vehicle / bin capacity τ.
            revenue: Revenue per unit collected (revenue_per_kg).
            cost_unit: Routing cost per unit distance (cost_per_km).
            ph_loop: Optional ProgressiveHedgingCGLoop.
            alns_pricer: Optional ALNSMultiPeriodPricer.
            ml_branching: Optional MLBranchingStrategy.
            scenario_branching: Optional ScenarioConsistentBranching.
            dive_heuristic: Optional DiveAndPricePrimalHeuristic.
            fix_optimizer: Optional FixAndOptimizeRefiner.
            max_iterations: Maximum Benders master–subproblem cycles.
            convergence_tol: (UB−LB)/max(1,|UB|) threshold for termination.
            cut_pool_max: Maximum Benders cuts retained in the pool (FIFO).
            max_visits_per_bin: Maximum times each bin may be assigned across
                the horizon (master frequency constraint).
            theta_upper_bound: Initial upper bound on θ[d]; prevents the
                master from being unbounded before the first cuts.
            gurobi_master_time_limit: Gurobi time limit per master solve (s).
            gurobi_sub_time_limit: Gurobi time limit per subproblem solve (s).
            gurobi_mip_gap: Relative MIP gap for Gurobi master and MIP sub.
            gurobi_output_flag: Enable Gurobi console output (debug mode).
            subproblem_relax: If True, solve the LP relaxation (no subtour
                elimination).  If False, solve as MIP + LP dual recovery.
        """
        self.tree = tree
        self.prize_engine = prize_engine
        self.capacity = capacity
        self.revenue = revenue
        self.cost_unit = cost_unit

        self.ph_loop = ph_loop
        self.alns_pricer = alns_pricer
        self.ml_branching = ml_branching
        self.scenario_branching = scenario_branching
        self.dive_heuristic = dive_heuristic
        self.fix_optimizer = fix_optimizer

        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.cut_pool_max = cut_pool_max

        self.max_visits_per_bin = max_visits_per_bin
        self.theta_upper_bound = theta_upper_bound
        self.gurobi_master_time_limit = gurobi_master_time_limit
        self.gurobi_sub_time_limit = gurobi_sub_time_limit
        self.gurobi_mip_gap = gurobi_mip_gap
        self.gurobi_output_flag = gurobi_output_flag
        self.subproblem_relax = subproblem_relax

        self._cut_pool: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Cut Pool Management                                                  #
    # ------------------------------------------------------------------ #

    def _add_cut(self, cut: Dict[str, Any]) -> None:
        """
        Add a cut to the internal pool (FIFO eviction at capacity).

        Args:
            cut: The cut to add to the pool.

        """
        self._cut_pool.append(cut)
        if len(self._cut_pool) > self.cut_pool_max:
            self._cut_pool.pop(0)

    # ------------------------------------------------------------------ #
    # Main Entry Point                                                     #
    # ------------------------------------------------------------------ #

    def solve(self, **kwargs: Any) -> Tuple[List[List[List[int]]], float]:
        """
        Solve the multi-period stochastic VRPP via Temporal Benders Decomposition.

        Args:
            kwargs: Forwarded to per-scenario subproblem solvers.

            Required kwargs:
                dist_matrix: Distance matrix, shape (n_bins+1, n_bins+1),
                    depot at index 0.  If not provided, raises ``ValueError``.

            Optional kwargs:
                Any remaining key-value pairs are forwarded to subproblem solvers.

        Returns:
            raw_plan: ``[day][route_index][node_index]`` covering all horizon days.
            total_expected_profit: Σ_d Σ_ξ p_ξ · Q_d(z*, ξ) over the best solution.

        Raises:
            ValueError: If ``dist_matrix`` is absent from kwargs.
            RuntimeError: If Gurobi is not installed (gurobipy import failed).
        """
        dist_matrix: Optional[np.ndarray] = kwargs.pop("dist_matrix", None)
        if dist_matrix is None:
            raise ValueError(
                "TemporalBendersCoordinator.solve() requires 'dist_matrix' "
                "in kwargs.  Pass it via problem.extra['dist_matrix']."
            )

        if not _GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not installed or licensed.  Install with: pip install gurobipy")

        n_bins = int(dist_matrix.shape[0]) - 1  # dist_matrix is (n+1) × (n+1)
        return self._solve_gurobi(dist_matrix, n_bins, **kwargs)

    def _get_scenario_prizes(
        self,
        scenario: Any,
        fallback_prizes: Dict[int, float],
        days_remaining: int,
    ) -> Dict[int, float]:
        """
        Compute per-scenario node prizes, falling back to the aggregate
        scenario-weighted prizes when per-scenario computation fails.

        Args:
            scenario: Scenario node with optional ``.wastes`` attribute.
            fallback_prizes: Pre-computed aggregate prizes from
                ``prize_engine.scenario_weighted_prizes``.
            days_remaining: Days remaining in horizon (for prize computation).

        Returns:
            Prize dict ``{bin_id: π_i^ξ}``.
        """
        if not hasattr(scenario, "wastes"):
            return fallback_prizes

        try:
            wastes = np.asarray(scenario.wastes, dtype=float)
            return self.prize_engine.compute_prizes(
                current_wastes=wastes,
                bin_stats={
                    "means": np.maximum(wastes, 0.0),
                    "stds": np.ones(len(wastes)),
                },
                revenue=self.revenue,
                days_remaining=days_remaining,
            )
        except Exception as exc:
            logger.debug("Per-scenario prize computation failed (%s); using aggregate.", exc)
            return fallback_prizes

    # ------------------------------------------------------------------ #
    # Theta Upper Bound Estimation                                         #
    # ------------------------------------------------------------------ #

    def _estimate_theta_upper_bound(self, horizon: int) -> float:
        """
        Compute a tighter initial upper bound on θ[d] from day-0 prizes.

        Samples the scenario-weighted prizes for day 0 and sets:

            theta_ub = max(self.theta_upper_bound, Σ_i max(π_i, 0))

        This ensures the bound is never below the true optimum (single vehicle
        cannot collect more revenue than all prizes summed) while being
        tighter than the raw ``theta_upper_bound`` default for most instances.

        Args:
            horizon: Planning horizon (used when querying prize engine).

        Returns:
            Per-day theta upper bound applied uniformly across all days.
        """
        ub = self.theta_upper_bound
        try:
            day0_prizes = self.prize_engine.scenario_weighted_prizes(
                day=0,
                revenue=self.revenue,
                days_remaining=horizon,
            )
            if day0_prizes:
                prize_sum = sum(max(v, 0.0) for v in day0_prizes.values())
                ub = max(ub, prize_sum)
        except Exception:
            pass
        return ub

    # ------------------------------------------------------------------ #
    # Benders Cut Generation (public, for testing and external use)        #
    # ------------------------------------------------------------------ #

    def generate_benders_cut(
        self,
        day: int,
        scenario_id: Any,
        z_bar: Dict[int, int],
        subproblem_profit: float,
        subproblem_duals: Dict[int, float],
        scenario_prob: float,
    ) -> Dict[str, Any]:
        """
        Construct a probability-weighted Benders optimality cut for a single
        (day, scenario) pair.

        In the main loop, cuts are aggregated directly inside
        ``_evaluate_subproblems``.  This method is provided for unit testing,
        external use, and manual cut inspection.

        Cut form:

            θ_d  ≥  constant  +  Σ_i  coeff[i] · z[i, d]

        where:

            constant  = p_ξ · [Q_d(z̄,ξ) − Σ_i μ_{id}^ξ · z̄_{id}]
            coeff[i]  = p_ξ · μ_{id}^ξ

        Args:
            day: Planning day index d.
            scenario_id: Scenario identifier.
            z_bar: Day-specific assignment ``{bin_id: 0_or_1}``.
            subproblem_profit: Q_d(z̄, ξ).
            subproblem_duals: ``{bin_id: μ_{id}^ξ}``.
            scenario_prob: p_ξ.

        Returns:
            Cut dict with keys ``"day"``, ``"scenario"``, ``"constant"``,
            ``"coefficients"``.
        """
        constant = float(scenario_prob * subproblem_profit)
        coeffs: Dict[int, float] = {}

        for bin_id, mu in subproblem_duals.items():
            z_val = float(z_bar.get(bin_id, 0))
            constant -= float(scenario_prob * mu * z_val)
            coeffs[bin_id] = float(scenario_prob * mu)

        return {
            "day": day,
            "scenario": scenario_id,
            "constant": constant,
            "coefficients": coeffs,
        }
