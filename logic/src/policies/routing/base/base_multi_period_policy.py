"""
Base Multi-Period Routing Policy.

Provides a template for routing policies that perform T-period optimization
over a scenario tree, rather than myopic single-day routing.

Attributes:
    BaseMultiPeriodRoutingPolicy: Policy for multi-period stochastic routing.

Example:
    >>> from logic.src.policies.routing.base.base_multi_period_policy import BaseMultiPeriodRoutingPolicy
    >>> # subclass and implement _run_multi_period_solver
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from logic.src.interfaces.context.multi_day_context import MultiDayContext
from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.interfaces.context.search_context import SearchContext
from logic.src.interfaces.context.solution_context import SolutionContext
from logic.src.pipeline.simulations.bins.prediction import ScenarioTree

if TYPE_CHECKING:
    pass

from .base_routing_policy import BaseRoutingPolicy


class BaseMultiPeriodRoutingPolicy(BaseRoutingPolicy):
    """
    Base class for policies optimized for stochastic multi-period routing (SIRP).

    This class provides utilities for:
    - Horizon management (T days).
    - Scenario tree traversal.
    - Future mandatory node prediction.
    - Overflow (stockout) penalty calculation.

    Attributes:
        horizon (int): Planning horizon in days (T).
        stockout_penalty (float): Penalty per unit of waste overflow (>100%).

    """

    def __init__(self, config: Any = None):
        """
        Initializes the BaseMultiPeriodRoutingPolicy.

        Args:
            config: Optional configuration object.
        """
        super().__init__(config)
        self.horizon: int = getattr(self.config, "horizon", 7)
        self.stockout_penalty: float = getattr(self.config, "stockout_penalty", 500.0)

    def _predict_mandatory_nodes_for_horizon(
        self, tree: ScenarioTree, initial_mandatory: List[int]
    ) -> Dict[int, List[int]]:
        """Predict mandatory nodes for each day in the horizon based on the tree.

        Args:
            tree: Scenario tree representing stochastic waste growth.
            initial_mandatory: List of nodes mandatory for the current day (day 0).

        Returns:
            Dictionary mapping day offsets (0 to T) to lists of mandatory bin IDs.
        """

        mandatory_map: Dict[int, List[int]] = {0: initial_mandatory}

        # We traverse the main scenario path (or all branches) to identify
        # when bins are expected to overflow (>100%).
        for t in range(1, tree.horizon + 1):
            scenarios = tree.get_scenarios_at_day(t)
            # For simplicity, we mark a node as mandatory if it overflows in ANY scenario at day t,
            # but hasn't been collected in previous days t' < t.
            # Real solvers might handle this more robustly.
            day_mandatory = []
            for scenario in scenarios:
                overflow_bins = np.where(scenario.wastes >= 100.0)[0] + 1  # 1-based
                for b_id in overflow_bins:
                    if b_id not in day_mandatory:
                        # Check if it was already mandatory in a previous day
                        already_seen = False
                        for prev_t in range(t):
                            if b_id in mandatory_map.get(prev_t, []):
                                already_seen = True
                                break
                        if not already_seen:
                            day_mandatory.append(int(b_id))

            mandatory_map[t] = day_mandatory

        return mandatory_map

    @abstractmethod
    def _run_multi_period_solver(
        self,
        problem: "ProblemContext",
        multi_day_ctx: Optional[MultiDayContext],
    ) -> Tuple["SolutionContext", List[List[List[int]]], Dict[str, Any]]:
        """
        Solve the T-period optimization problem.

        Args:
            problem:       Fully typed problem descriptor for today.
            multi_day_ctx: Current multi-day state (may be None on day 0).

        Returns:
            Tuple of:
                solution   – SolutionContext for today's routes only.
                full_plan  – List[day][vehicle][node] for the whole horizon
                             (to be stored in MultiDayContext.full_plan_snapshot).
                metadata   – Solver telemetry dict.
        """
        pass

    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """Dummy implementation of the single-period solver.

        Multi-period policies override the execute() method directly (or rely on
        the BaseMultiPeriodRoutingPolicy override) to call _run_multi_period_solver,
        so this method is never invoked in the SIRP workflow.

        Args:
            sub_dist_matrix: Subset distance matrix.
            sub_wastes: Local wastes dict.
            capacity: Vehicle capacity.
            revenue: Revenue per unit.
            cost_unit: Cost per distance unit.
            values: Merged config values.
            mandatory_nodes: Local indices of mandatory nodes.
            kwargs: Additional keyword arguments.


        Returns:
            Tuple of (empty_routes, 0.0 profit, 0.0 cost).
        """

        return [], 0.0, 0.0

    def execute(
        self, **kwargs: Any
    ) -> Tuple[Union[List[int], List[List[int]]], float, float, Optional[SearchContext], Optional[MultiDayContext]]:
        """Executes the multi-period routing policy.

        This implementation extends the base execution by building a `ProblemContext`,
        invoking the multi-period solver, and updating the `MultiDayContext` with
        the full plan snapshot.

        Args:
            kwargs: Keyword arguments containing context parameters:

                - area (str): Geographic area name.
                - waste_type (str): Waste type (e.g., "plastic").
                - config (dict): Policy configuration overrides.
                - multi_day_context (MultiDayContext): Persistence across days.
                - search_context (SearchContext): Metrics and telemetry ledger.
                - bins: Bins state object.
                - distance_matrix: NxN distance matrix.

        Returns:
            A tuple of (tour, cost, profit, search_context, multi_day_context).
        """

        from logic.src.interfaces.context.problem_context import ProblemContext

        # 1. Load area params (unchanged — uses existing _load_area_params)
        capacity, revenue, cost_unit, values = self._load_area_params(
            kwargs.get("area", "Rio Maior"),
            kwargs.get("waste_type", "plastic"),
            kwargs.get("config", {}),
        )

        multi_day_ctx: Optional[MultiDayContext] = kwargs.get("multi_day_context")
        search_ctx: Optional[SearchContext] = kwargs.get("search_context")

        if search_ctx is None:
            search_ctx = SearchContext.initialize()

        # 2. Empty-mandatory ruling (D3): same contract as the single-period
        # BaseRoutingPolicy.execute — an empty mandatory set yields the empty
        # tour [0, 0] at zero cost/profit instead of running the solver.
        early_result = self._validate_mandatory(kwargs.get("mandatory"))
        if early_result is not None:
            tour, cost, profit = early_result
            return tour, cost, profit, search_ctx, multi_day_ctx

        # 3. Build typed problem descriptor
        problem = ProblemContext.from_kwargs(kwargs, capacity, revenue, cost_unit)

        # 4. Solve
        solution, full_plan, solver_metadata = self._run_multi_period_solver(problem, multi_day_ctx)

        # Merge telemetry into the ledger
        from logic.src.interfaces.context.search_context import SearchPhase, merge_context

        search_ctx = merge_context(search_ctx, phase=SearchPhase.ROUTING, construction_metrics=solver_metadata)

        # 5. Update MultiDayContext with full plan snapshot
        if multi_day_ctx is not None:
            multi_day_ctx = multi_day_ctx.update(
                full_plan_snapshot=full_plan,
                extra={**multi_day_ctx.extra, **solver_metadata},
            )

        # 6. Return in the existing 5-tuple format
        tour = solution.to_flat_tour()
        cost = solution.total_cost
        profit = solution.total_profit

        return tour, cost, profit, search_ctx, multi_day_ctx
