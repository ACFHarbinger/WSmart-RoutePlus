"""
Module documentation.

Attributes:
    ABPCHGPolicy: The Adaptive Branch-and-Price-and-Cut with Heuristic Guidance policy implementation.

Examples:
    >>> from logic.src.policies.routing.adaptive_branch_and_price_and_cut_with_heuristic_guidance.policy_abpc_hg import ABPCHGPolicy
    >>> policy = ABPCHGPolicy()
    >>> policy.run(problem_context)
"""

from typing import Any, Dict, List, Optional, Tuple, Type

from logic.src.configs.policies.abpc_hg import ABPCHGConfig
from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.policies.helpers.solvers_and_matheuristics import RCSPPSolver
from logic.src.policies.routing.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
from logic.src.policies.routing.base.factory import RoutingAlgorithmRegistry

from .alns_pricer import ALNSMultiPeriodPricer
from .dive_and_price import DiveAndPricePrimalHeuristic
from .fix_and_optimize import FixAndOptimizeRefiner
from .ml_branching import MLBranchingStrategy
from logic.src.configs.policies.abpc_hg import ABPCHGConfig
from .progressive_hedging_cg import ProgressiveHedgingCGLoop
from .scenario_branching import ScenarioConsistentBranching
from .scenario_prize_engine import ScenarioPrizeEngine
from .temporal_benders import TemporalBendersCoordinator


@GlobalRegistry.register(
    PolicyTag.MATHEURISTIC,
    PolicyTag.DECOMPOSITION,
    PolicyTag.REINFORCEMENT_LEARNING,
    PolicyTag.MULTI_PERIOD,
    PolicyTag.PROFIT_AWARE,
)
@RoutingAlgorithmRegistry.register("abpc_hg")
class ABPCHGPolicy(BaseMultiPeriodRoutingPolicy):
    """
    Adaptive Branch-and-Price-and-Cut with Heuristic Guidance (ABPC-HG).

    Overview
    --------
    ABPC-HG is a hybrid exact–matheuristic policy for Stochastic Multi-Period
    Vehicle Routing Problems with Profits (SMPVRPP).  It wraps a rigorous
    Temporal Benders Decomposition over the full planning horizon with a suite
    of scenario-aware heuristic components that inject structure from the
    ScenarioTree at every level of the exact BPC hierarchy.

    The policy assumes the ProblemContext carries a populated ScenarioTree and
    raises ``ValueError`` if it is absent.

    Pipeline
    --------
    Execution proceeds in four phases:

    **Phase 1 — Scenario Prize Augmentation**
        ``ScenarioPrizeEngine`` computes Bellman-style node prizes π_i
        encoding both immediate fill revenue and the discounted opportunity
        cost of deferral, conditioned on the scenario-tree fill distribution
        and the Gamma-based overflow predictor:

            π_i = w_{i,d} · R
                + γ · E_ξ[V_future(i, not visited)]
                − γ · E_ξ[V_future(i, visited)]

    **Phase 2 — Root-Node CG with Non-Anticipativity (Progressive Hedging)**
        ``ProgressiveHedgingCGLoop`` maintains one Restricted Master Problem
        per scenario ξ and coordinates them toward a common solution by
        penalising deviations from the consensus variable x̄ via an
        augmented-Lagrangian proximal term:

            c'_{k,ξ}  +=  −ρ_eff · (x_k − x̄_k)²

        where ρ_eff = ρ₀ · (1 + Var_ξ[π_i^ξ]) is set dynamically per variable.

    **Phase 3 — Temporal Benders Loop**
        ``TemporalBendersCoordinator`` iteratively tightens a master-problem
        lower bound through probability-weighted optimality cuts
        θ_d ≥ p_ξ · [Q_d(z̄,ξ) + Σ_i μ_{id}^ξ (z_{id} − z̄_{id})].

        Within each Benders iteration:

        *Heuristic pricing* (``ALNSMultiPeriodPricer``): scenario-overflow-
        driven destroy + scenario-aware insertion repair operators generate
        candidate columns, falling back to exact ESPPRC when no improving
        column is found.

        *Primal bound* (``DiveAndPricePrimalHeuristic``): greedily fixes the
        most scenario-consistent fractional root columns
        (score = λ_k · consensus_k) with soft big-M recovery on infeasibility.

        *Branching* (``MLBranchingStrategy`` + ``ScenarioConsistentBranching``):
        GNN-based imitation-learning surrogate for strong branching, falling
        back to reliability-branching pseudo-costs; the scenario-consistent
        rule maximises |consensus(g_i) − 0.5| across fractional variables.

    **Phase 4 — Fix-and-Optimize Polish**
        ``FixAndOptimizeRefiner`` iteratively unfixes targeted bin clusters
        (selected by overflow urgency or cross-scenario fill variance, guarded
        by a Jaccard-similarity tabu list) and hands each corridor to the
        exact BPC engine for re-optimisation.

    Configuration
    -------------
    All tunable parameters are collected in ``ABPCHGConfig``.  The full list
    with documentation and calibration guidance is available there.  A minimal
    config need only specify ``gamma``; all other fields fall back to defaults.

    External Injection
    ------------------
    An exact RCSPP solver (``RCSPPSolver``) may be passed through
    ``ProblemContext.extra["exact_pricer"]`` to serve as the ALNS fallback
    oracle.  If absent, ALNS-only pricing is used without an exact fallback.

    A trained GNN branching model may be injected post-construction via:
        ``policy.ml_branching_model = <your model>``
    and will be picked up on the next ``_run_multi_period_solver`` call.

    Attributes:
    ----------
    params : ABPCHGConfig
        Fully resolved configuration instance built from the supplied config.
    gamma : float
        Convenience alias for ``params.gamma``.
    ml_branching_model : Any or None
        Optional trained GNN model injected externally; ``None`` causes
        ``MLBranchingStrategy`` to fall back to reliability branching.
    """

    def __init__(self, config: Optional[ABPCHGConfig] = None):
        """Initialize ABPCHGPolicy.

        Args:
            config: Configuration for the policy.
        """
        super().__init__(config)
        self.params: ABPCHGConfig = self._effective_config(config) if config is not None else ABPCHGConfig()
        self.gamma: float = self.params.gamma
        # Externally injectable trained GNN model for MLBranchingStrategy.
        # Set to a loaded model before solving to activate learned branching.
        self.ml_branching_model: Any = None

    @classmethod
    def _config_class(cls) -> Type[ABPCHGConfig]:
        """Return the configuration class for the policy.

        Returns:
            Type[ABPCHGConfig]: The configuration class for the policy.
        """
        return ABPCHGConfig

    def _get_config_key(self) -> str:
        """Return the configuration key for the policy.

        Returns:
            str: The configuration key for the policy.
        """
        return "abpc_hg"

    # ------------------------------------------------------------------ #
    # Component Factories                                                  #
    # ------------------------------------------------------------------ #

    def _build_prize_engine(self, tree: Any, capacity: float) -> ScenarioPrizeEngine:
        """Instantiate a ``ScenarioPrizeEngine`` with policy-level parameters.

        Args:
            tree: ScenarioTree for the current horizon.
            capacity: Bin / vehicle capacity τ, used as the overflow cap.

        Returns:
            Configured ``ScenarioPrizeEngine`` instance.
        """
        return ScenarioPrizeEngine(
            scenario_tree=tree,
            gamma=self.params.gamma,
            tau=capacity,
            overflow_penalty=self.params.overflow_penalty,
        )

    def _build_ph_loop(self, num_scenarios: int) -> ProgressiveHedgingCGLoop:
        """Instantiate the Progressive Hedging root-node CG loop.

        Args:
            num_scenarios: Total number of scenarios in the tree (denominator
                of the x̄ average update).

        Returns:
            Configured ``ProgressiveHedgingCGLoop`` instance.
        """
        return ProgressiveHedgingCGLoop(
            num_scenarios=num_scenarios,
            base_rho=self.params.ph_base_rho,
        )

    def _build_alns_pricer(self, exact_pricer: RCSPPSolver) -> ALNSMultiPeriodPricer:
        """Instantiate the ALNS heuristic pricing oracle with an exact fallback.

        Args:
            exact_pricer: Exact ``RCSPPSolver`` used when ALNS cannot produce
                a column with reduced cost > ``alns_rc_tolerance``.

        Returns:
            Configured ``ALNSMultiPeriodPricer`` instance.
        """
        return ALNSMultiPeriodPricer(
            exact_pricer=exact_pricer,
            rng_seed=self.params.seed if self.params.seed is not None else 42,
        )

    def _build_dive_heuristic(self) -> DiveAndPricePrimalHeuristic:
        """Instantiate the Dive-and-Price primal heuristic.

        Returns:
            Configured ``DiveAndPricePrimalHeuristic`` instance.
        """
        return DiveAndPricePrimalHeuristic(penalty_M=self.params.dive_penalty_M)

    def _build_fix_optimizer(self) -> FixAndOptimizeRefiner:
        """Instantiate the Fix-and-Optimize corridor refiner.

        Returns:
            Configured ``FixAndOptimizeRefiner`` instance with tabu list
            capacity and corridor width from ``params``.
        """
        return FixAndOptimizeRefiner(
            tabu_length=self.params.fo_tabu_length,
            max_unfix=self.params.fo_max_unfix,
        )

    def _build_ml_branching(self) -> MLBranchingStrategy:
        """Instantiate the ML / reliability branching strategy.

        Uses ``self.ml_branching_model`` if set (external injection);
        otherwise constructs a model-free instance that falls back to
        reliability branching throughout.

        Returns:
            Configured ``MLBranchingStrategy`` instance.
        """
        return MLBranchingStrategy(
            model=self.ml_branching_model,
            reliability_c=self.params.ml_reliability_c,
        )

    def _build_scenario_branching(self) -> ScenarioConsistentBranching:
        """Instantiate the scenario-consistent branching rule.

        Returns:
            Configured ``ScenarioConsistentBranching`` instance.
        """
        return ScenarioConsistentBranching(base_threshold=self.params.sc_consensus_threshold)

    def _build_coordinator(
        self,
        tree: Any,
        prize_engine: ScenarioPrizeEngine,
        capacity: float,
        revenue: float,
        cost_unit: float,
        num_scenarios: int,
        exact_pricer: Optional[RCSPPSolver] = None,
    ) -> TemporalBendersCoordinator:
        """
        Assemble and wire all pipeline components into a ``TemporalBendersCoordinator``.

        This is the single point where all ``_build_*`` factory methods are
        called and their outputs composed.  The resulting coordinator is a
        fully wired, self-contained solver ready to call ``.solve()``.

        Args:
            tree: Scenario tree for the planning horizon.
            prize_engine: Pre-built ``ScenarioPrizeEngine`` instance.
            capacity: Vehicle / bin capacity τ.
            revenue: Revenue per unit of waste collected.
            cost_unit: Routing cost per unit distance.
            num_scenarios: Leaf scenario count; denominator of the PH x̄ average.
            exact_pricer: Optional exact ESPPRC solver for ALNS fallback.
                If ``None``, the ALNS pricer is not constructed and pricing
                falls back entirely to the coordinator's subproblem stub.

        Returns:
            Fully wired ``TemporalBendersCoordinator``.
        """
        alns_pricer = self._build_alns_pricer(exact_pricer) if exact_pricer is not None else None

        return TemporalBendersCoordinator(
            tree=tree,
            prize_engine=prize_engine,
            capacity=capacity,
            revenue=revenue,
            cost_unit=cost_unit,
            ph_loop=self._build_ph_loop(num_scenarios),
            alns_pricer=alns_pricer,
            ml_branching=self._build_ml_branching(),
            scenario_branching=self._build_scenario_branching(),
            dive_heuristic=self._build_dive_heuristic(),
            fix_optimizer=self._build_fix_optimizer(),
            max_iterations=self.params.benders_max_iterations,
            convergence_tol=self.params.benders_convergence_tol,
            cut_pool_max=self.params.benders_cut_pool_max,
            max_visits_per_bin=self.params.max_visits_per_bin,
            theta_upper_bound=self.params.theta_upper_bound,
            gurobi_master_time_limit=self.params.gurobi_master_time_limit,
            gurobi_sub_time_limit=self.params.gurobi_sub_time_limit,
            gurobi_mip_gap=self.params.gurobi_mip_gap,
            gurobi_output_flag=self.params.gurobi_output_flag,
            subproblem_relax=self.params.subproblem_relax,
        )

    # ------------------------------------------------------------------ #
    # Main Entry Point                                                     #
    # ------------------------------------------------------------------ #

    def _run_multi_period_solver(
        self,
        problem: ProblemContext,
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple[SolutionContext, List[List[List[int]]], Dict[str, Any]]:
        """
        Execute the full ABPC-HG pipeline over the multi-day planning horizon.

        Resolves all problem-level parameters from ``ProblemContext``,
        constructs the component graph via the ``_build_*`` factory methods,
        and delegates execution to ``TemporalBendersCoordinator.solve()``.

        Args:
            problem: ProblemContext carrying the current state, including:
                - ``scenario_tree``: populated ``ScenarioTree`` (required).
                - ``capacity``: bin / vehicle capacity τ.
                - ``revenue_per_kg``: revenue coefficient R.
                - ``cost_per_km``: routing cost coefficient.
                - ``extra``: optional dict of keyword arguments forwarded to
                  sub-solvers; may include ``"exact_pricer"`` (``RCSPPSolver``)
                  to activate exact ESPPRC fallback in ALNS pricing.
            multi_day_ctx: Optional ``MultiDayContext`` for cross-day state
                propagation (e.g. rolling-horizon warm starts, column pool
                carryover).  May be ``None`` at day 0 or in single-day runs.

        Returns:
            today_solution : SolutionContext
                Standardised solution for day 0 built from ``raw_plan[0][0]``.
            full_plan : List[List[List[int]]]
                Full collection plan ``[day][route_index][node_index]``
                spanning the entire horizon.
            stats : Dict[str, Any]
                Execution metadata with keys:

                ``"policy"``
                    Policy identifier ``"abpc_hg"``.
                ``"expected_profit"``
                    Total probability-weighted profit Σ_d Σ_ξ p_ξ · Q_d(z*,ξ).
                ``"benders_iterations"``
                    Number of Benders iterations executed (upper bound;
                    may be fewer if convergence was achieved early).
                ``"params"``
                    The ``ABPCHGConfig`` instance used for this solve,
                    enabling reproducibility and experiment logging.

        Raises:
            ValueError: If ``problem.scenario_tree`` is ``None``.  The entire
                ABPC-HG pipeline is conditioned on scenario-tree availability;
                single-scenario / deterministic problems should use a
                degenerate one-scenario tree rather than passing ``None``.
        """
        tree = problem.scenario_tree
        if tree is None:
            raise ValueError(
                "ABPC-HG requires a ScenarioTree in ProblemContext.  "
                "For deterministic problems, wrap the single scenario in a "
                "one-leaf ScenarioTree."
            )

        capacity = problem.capacity
        revenue = problem.revenue_per_kg
        cost_unit = problem.cost_per_km

        # Count leaf scenarios for PH x̄ denominator
        num_scenarios = len(tree.get_leaves()) if hasattr(tree, "get_leaves") else 1

        # Resolve dist_matrix: check ProblemContext attribute first, then extra.
        # The distance matrix must be provided as (n_bins+1 × n_bins+1) with
        # the depot at index 0.  Raise early with a clear message if absent.
        dist_matrix = getattr(problem, "dist_matrix", None)
        if dist_matrix is None:
            dist_matrix = problem.extra.pop("dist_matrix", None)
        if dist_matrix is None:
            raise ValueError(
                "ABPC-HG requires a distance matrix.  "
                "Provide it as problem.dist_matrix or problem.extra['dist_matrix']."
            )
        # Ensure dist_matrix is in extra so it reaches coordinator.solve()
        problem.extra["dist_matrix"] = dist_matrix

        # Resolve optional exact pricer injected through problem extras
        exact_pricer: Optional[RCSPPSolver] = problem.extra.pop("exact_pricer", None)

        # Build scenario prize engine
        prize_engine = self._build_prize_engine(tree, capacity)

        # Assemble the fully wired coordinator
        coordinator = self._build_coordinator(
            tree=tree,
            prize_engine=prize_engine,
            capacity=capacity,
            revenue=revenue,
            cost_unit=cost_unit,
            num_scenarios=num_scenarios,
            exact_pricer=exact_pricer,
        )

        raw_plan, total_expected_profit = coordinator.solve(**problem.extra)

        today_route = raw_plan[0][0] if raw_plan and raw_plan[0] else []
        sol = SolutionContext.from_problem(problem, today_route)

        return (
            sol,
            raw_plan,
            {
                "policy": "abpc_hg",
                "expected_profit": total_expected_profit,
                "benders_iterations": coordinator.max_iterations,
                "params": self.params,
            },
        )
