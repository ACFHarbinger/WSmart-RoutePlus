r"""Horizon-level support methods for the ALNS-IPO solver.

Moved unchanged from :mod:`alns_ipo`: this module holds the mixin that
builds the initial T-day horizon solution, evaluates horizon cost and
profit, builds the forward-looking repair closure, and applies
single-day operators inside the horizon search.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np

from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
from logic.src.configs.policies.routing.meta_heuristics.alns_ipo import ALNSIPOConfig
from logic.src.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators import (
    build_greedy_routes,
    forward_looking_insertion,
    greedy_insertion,
    greedy_profit_insertion,
)


class ALNSHorizonSupportMixin:
    r"""Horizon solution construction and evaluation for the ALNS-IPO solver."""

    # Provided by ALNSSolverIPO.__init__ and ALNSSolver.__init__; annotations only.
    horizon: int
    dist_matrix: np.ndarray
    wastes: Dict[int, float]
    capacity: float
    R: float
    C: float
    mandatory_nodes: Optional[List[int]]
    random: random.Random
    calculate_cost: Callable[[List[List[int]]], float]
    _scenario_tree: Any
    stockout_penalty: float
    forward_looking_depth: int
    inventory_lambda: float
    destroy_ops: List[Callable[..., Any]]
    params: ALNSConfig
    profit_aware_operators: bool

    # ------------------------------------------------------------------
    # Helper: build initial horizon solution
    # ------------------------------------------------------------------

    def build_initial_horizon_solution(self) -> List[List[List[int]]]:
        """Build an initial T-day horizon solution using the parent greedy heuristic.

        Returns:
            A full T-day initial plan.
        """
        horizon: List[List[List[int]]] = []
        for _ in range(self.horizon):
            day_routes = build_greedy_routes(
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                mandatory_nodes=self.mandatory_nodes,
                rng=self.random,
            )
            horizon.append(day_routes)
        return horizon

    # ------------------------------------------------------------------
    # Horizon cost evaluation
    # ------------------------------------------------------------------

    def calculate_horizon_cost(self, horizon_routes: List[List[List[int]]]) -> float:
        """Compute total routing cost across the horizon.

        Args:
            horizon_routes: Full T-day solution to evaluate.

        Returns:
            Total routing cost.
        """
        total = 0.0
        for day_routes in horizon_routes:
            total += self.calculate_cost(day_routes)
        return total

    def calculate_horizon_profit(self, horizon_routes: List[List[List[int]]]) -> float:
        """Compute total profit (revenue − routing cost) across the horizon.

        Args:
            horizon_routes: Full T-day solution to evaluate.

        Returns:
            Total net profit.
        """
        total_cost = self.calculate_horizon_cost(horizon_routes)
        total_rev = sum(
            self.wastes.get(node, 0.0) * self.R
            for day_routes in horizon_routes
            for route in day_routes
            for node in route
        )
        return total_rev - total_cost

    def _make_fl_repair(self) -> Callable:
        """Build the forward-looking repair operator closure.

        Returns:
            The repair operator function.
        """

        def _fl(horizon_routes: List[List[List[int]]], removed: Any, _noise: float = 0.0) -> List[List[List[int]]]:
            return forward_looking_insertion(
                horizon_routes=horizon_routes,
                removed=removed,
                dist_matrix=self.dist_matrix,
                wastes=self.wastes,
                capacity=self.capacity,
                R=self.R,
                C=self.C,
                scenario_tree=self._scenario_tree,
                stockout_penalty=self.stockout_penalty,
                look_ahead_days=self.forward_looking_depth,
                lambda_inventory=self.inventory_lambda,
            )

        return _fl

    def _apply_single_day_ops(self, day_routes: List[List[int]], d_idx: int) -> Tuple[List[List[int]], List[int]]:
        """Apply a single-day destroy operator and greedily repair.

        Args:
            day_routes: Routes for the selected day.
            d_idx: Index of the destroy operator to apply.

        Returns:
            Tuple containing (repaired_routes, removed_nodes).
        """
        destroy_op = self.destroy_ops[d_idx]
        n_nodes = sum(len(r) for r in day_routes)
        if n_nodes == 0:
            return day_routes, []

        ipo_params = cast(ALNSIPOConfig, self.params)
        lower = min(n_nodes, ipo_params.min_removal)
        upper = min(n_nodes, self.params.max_removal_cap, max(lower, int(n_nodes * ipo_params.max_removal_pct)))
        upper = max(upper, lower)
        n_rem = self.random.randint(lower, upper)

        new_routes, removed = destroy_op([r[:] for r in day_routes], n_rem)

        # Repair with greedy (single-day)
        common_kw: Dict = dict(
            dist_matrix=self.dist_matrix,
            wastes=self.wastes,
            capacity=self.capacity,
            mandatory_nodes=self.mandatory_nodes,
            expand_pool=current_expand_pool(),
        )
        if self.profit_aware_operators:
            repaired = greedy_profit_insertion(new_routes, removed, R=self.R, C=self.C, **common_kw)
        else:
            repaired = greedy_insertion(new_routes, removed, **common_kw)

        return repaired, removed
