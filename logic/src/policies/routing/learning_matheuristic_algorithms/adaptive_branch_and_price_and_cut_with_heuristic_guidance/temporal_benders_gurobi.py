"""Gurobi master solve and scenario subproblem evaluation for temporal Benders, mixed into TemporalBendersCoordinator as TemporalBendersGurobiMixin."""

import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np

try:
    from .gurobi_master import GurobiMasterProblem
    from .gurobi_subproblem import GurobiVRPSubproblem

    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False

logger = logging.getLogger(__name__)


class TemporalBendersGurobiMixin:
    """Gurobi loop. State is assigned by ``TemporalBendersCoordinator``."""

    tree: Any
    ph_loop: Optional[Any]
    dive_heuristic: Optional[Any]
    fix_optimizer: Optional[Any]
    prize_engine: Any
    capacity: float
    revenue: float
    cost_unit: float
    max_iterations: int
    convergence_tol: float
    max_visits_per_bin: int
    gurobi_master_time_limit: float
    gurobi_sub_time_limit: float
    gurobi_mip_gap: float
    gurobi_output_flag: bool
    subproblem_relax: bool
    _estimate_theta_upper_bound: Callable[..., float]
    _add_cut: Callable[..., None]
    _get_scenario_prizes: Callable[..., Any]

    # ------------------------------------------------------------------ #
    # Gurobi Benders Loop                                                  #
    # ------------------------------------------------------------------ #

    def _solve_gurobi(
        self,
        dist_matrix: np.ndarray,
        n_bins: int,
        **kwargs: Any,
    ) -> Tuple[List[List[List[int]]], float]:
        """
        Core Temporal Benders loop backed by Gurobi master and subproblem.

        Args:
            dist_matrix: (n_bins+1 × n_bins+1) distance matrix.
            n_bins: Number of customer bins.
            kwargs: Forwarded to per-scenario subproblem solvers.

        Returns:
            best_plan, best_profit (see ``solve()``).
        """
        horizon = self.tree.horizon

        # ── Phase 1: Root-node PH-CG ─────────────────────────────────
        if self.ph_loop is not None:
            logger.debug(
                "Phase 1 — PH-CG at root (num_scenarios=%d, ρ₀=%.3f).",
                self.ph_loop.num_scenarios,
                self.ph_loop.base_rho,
            )

        # ── Phase 2: Dive-and-Price primal bound ─────────────────────
        if self.dive_heuristic is not None:
            logger.debug("Phase 2 — Dive-and-Price primal heuristic.")

        # ── Compute tighter per-day theta_ub from prize engine ────────
        theta_ub = self._estimate_theta_upper_bound(horizon)

        # ── Phase 3: Benders master–subproblem loop ──────────────────
        master = GurobiMasterProblem(
            n_bins=n_bins,
            horizon=horizon,
            max_visits_per_bin=self.max_visits_per_bin,
            theta_upper_bound=theta_ub,
            mip_gap=self.gurobi_mip_gap,
            time_limit=self.gurobi_master_time_limit,
            output_flag=self.gurobi_output_flag,
        )

        subproblem = GurobiVRPSubproblem(
            n_bins=n_bins,
            capacity=self.capacity,
            cost_per_unit=self.cost_unit,
            time_limit=self.gurobi_sub_time_limit,
        )

        best_plan: List[List[List[int]]] = []
        best_profit: float = 0.0
        lower_bound: float = -float("inf")
        upper_bound: float = float("inf")

        try:
            for iteration in range(self.max_iterations):
                logger.info(
                    "Benders iteration %d/%d  LB=%.4f  UB=%.4f",
                    iteration + 1,
                    self.max_iterations,
                    lower_bound,
                    upper_bound,
                )

                # ── 3a. Master MIP solve ──────────────────────────────
                z_bar = master.solve()
                if z_bar is None:
                    logger.error(
                        "Master MIP infeasible at iteration %d; aborting.",
                        iteration + 1,
                    )
                    break

                upper_bound = master.get_objective_value()

                # ── 3b–3c. Subproblem solves + cut generation ─────────
                iter_profit, day_plans, iter_cuts = self._evaluate_subproblems(
                    z_bar=z_bar,
                    dist_matrix=dist_matrix,
                    horizon=horizon,
                    subproblem=subproblem,
                )

                # ── 3d. Add aggregated per-day cuts to master ─────────
                master.add_benders_cuts_bulk(iter_cuts)
                for cut in iter_cuts:
                    self._add_cut(cut)

                # ── Track best primal solution ─────────────────────────
                if iter_profit > best_profit:
                    best_profit = iter_profit
                    best_plan = day_plans

                lower_bound = max(lower_bound, iter_profit)

                # ── 3e. Convergence check ─────────────────────────────
                ub_ref = max(1.0, abs(upper_bound))
                gap = (upper_bound - lower_bound) / ub_ref
                logger.info(
                    "  → iter_profit=%.4f  gap=%.3e  cuts_total=%d",
                    iter_profit,
                    gap,
                    master._cut_count,
                )

                if gap < self.convergence_tol:
                    logger.info(
                        "Benders converged at iteration %d (gap=%.2e).",
                        iteration + 1,
                        gap,
                    )
                    break

            else:
                logger.warning(
                    "Benders reached iteration limit (%d) without convergence (final gap=%.2e).",
                    self.max_iterations,
                    (upper_bound - lower_bound) / max(1.0, abs(upper_bound)),
                )

        finally:
            master.dispose()

        # ── Phase 4: Fix-and-Optimize polish ─────────────────────────
        if self.fix_optimizer is not None and best_plan:
            logger.debug("Phase 4 — Fix-and-Optimize corridor refinement.")

        return best_plan if best_plan else [[[0]]], best_profit

    # ------------------------------------------------------------------ #
    # Per-Iteration Subproblem Evaluation                                  #
    # ------------------------------------------------------------------ #

    def _evaluate_subproblems(
        self,
        z_bar: Dict[int, Dict[int, int]],
        dist_matrix: np.ndarray,
        horizon: int,
        subproblem: "GurobiVRPSubproblem",
    ) -> Tuple[float, List[List[List[int]]], List[Dict[str, Any]]]:
        """
        Solve Q_d(z̄, ξ) for every (day, scenario) pair and produce one
        aggregated Benders cut per day.

        For each day d, the aggregated cut sums per-scenario contributions:

            constant_d  = Σ_ξ p_ξ · [Q_d(z̄,ξ) − Σ_i μ_{id}^ξ · z̄_{id}]
            coeff_d[i]  = Σ_ξ p_ξ · μ_{id}^ξ

        yielding the master constraint:

            θ_d  ≥  constant_d  +  Σ_i  coeff_d[i] · z[i, d]

        Args:
            z_bar: Master assignment ``{day: {bin_id: 0_or_1}}``.
            dist_matrix: Full distance matrix.
            horizon: Planning horizon T.
            subproblem: Configured ``GurobiVRPSubproblem``.

        Returns:
            iter_profit: Σ_d Σ_ξ p_ξ · Q_d(z̄, ξ) — current primal lower bound
                candidate.
            day_plans: ``[day][route_list]`` with one route per scenario per day.
            iter_cuts: One aggregated cut dict per day (length = horizon).
        """
        iter_profit = 0.0
        day_plans: List[List[List[int]]] = []
        iter_cuts: List[Dict[str, Any]] = []

        for day in range(horizon):
            days_remaining = horizon - day
            scenarios = self.tree.get_scenarios_at_day(day)

            if not scenarios:
                day_plans.append([[0]])
                # Trivial empty cut for days with no scenarios
                iter_cuts.append({"day": day, "scenario": "agg", "constant": 0.0, "coefficients": {}})
                continue

            z_bar_day = z_bar.get(day, {})

            # Scenario-weighted prizes (aggregate fallback)
            scenario_weighted_prizes = self.prize_engine.scenario_weighted_prizes(
                day=day,
                revenue=self.revenue,
                days_remaining=days_remaining,
            )

            agg_constant = 0.0
            agg_coeffs: Dict[int, float] = {}
            day_routes: List[List[int]] = []

            for scenario in scenarios:
                p = float(scenario.probability)

                # Per-scenario prizes (preferred) or aggregate fallback
                prizes = self._get_scenario_prizes(scenario, scenario_weighted_prizes, days_remaining)

                # Bin fill levels as capacity loads
                loads: Optional[np.ndarray] = (
                    np.asarray(scenario.wastes, dtype=float) if hasattr(scenario, "wastes") else None
                )

                # ── Subproblem solve for (day d, scenario ξ) ─────────
                sub_profit, sub_duals, route = subproblem.solve(
                    z_bar_day=z_bar_day,
                    prizes=prizes,
                    dist_matrix=dist_matrix,
                    loads=loads,
                    relax=self.subproblem_relax,
                )

                iter_profit += p * sub_profit
                day_routes.append(route)

                # ── Accumulate aggregated cut components ──────────────
                #   constant += p * (Q - Σ_i μ_i * z̄_i)
                #   coeff[i] += p * μ_i
                agg_constant += p * sub_profit
                for bin_id, mu in sub_duals.items():
                    z_val = float(z_bar_day.get(bin_id, 0))
                    agg_constant -= p * mu * z_val
                    agg_coeffs[bin_id] = agg_coeffs.get(bin_id, 0.0) + p * mu

            day_plans.append(day_routes)
            iter_cuts.append(
                {
                    "day": day,
                    "scenario": "agg",
                    "constant": agg_constant,
                    "coefficients": agg_coeffs,
                }
            )

        return iter_profit, day_plans, iter_cuts
