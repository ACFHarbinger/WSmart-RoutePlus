r"""Stage 1 — Logic-Based Benders Decomposition (LBBD).

Decomposes the VRPP into:

Master problem (Gurobi MIP)
    Knapsack-style node-selection problem: choose *which* customers to visit
    subject to a profit-coverage lower bound.  The master accumulates Benders
    cuts from the sub-problem across iterations.

Sub-problem (routing oracle)
    Given the selected customer set Y ⊆ V, solve the routing problem to
    find the optimal route(s) covering Y.  The routing oracle is pluggable:
    'greedy' (fast), 'alns' (heuristic), or 'bpc' (exact).

Cut generation (Hooker 2000, Hooker & Ottosson 2003)
----------------------------------------------------
Three cut families are implemented:

1. No-good cuts  (always valid)
   Forbid the exact selection Y* found in the current iteration:
       Σ_{i ∈ Y*} y_i  ≤  |Y*| - 1

2. Optimality cuts  (valid when sub-problem is solved to optimality)
   If the routing cost for Y* is c*, any superset of Y* requires
   routing cost ≥ c*:
       θ  ≥  c*  −  M · (|Y*| − Σ_{i ∈ Y*} y_i)
   where θ is the routing cost proxy in the master.

3. Pareto-optimal (Magnanti-Wang 1981) cuts
   Strengthen optimality cuts by tightening M using the best-known
   primal bound π:
       θ  ≥  (c* − π)  ·  Σ_{i ∈ Y*} y_i  −  (c* − π) · |Y*| + π

All routes produced by the sub-problem are added to the shared RoutePool
for the SP-merge stage.

References:
    Hooker, J. N. (2000). Logic-Based Methods for Optimization. Wiley.
    Hooker, J. N., & Ottosson, G. (2003). Logic-based Benders decomposition.
        Mathematical Programming, 96(1), 33–60.
    Magnanti, T. L., & Wong, R. T. (1981). Accelerating Benders decomposition.
        Operations Research, 29(3), 464–484.
    Fachini, V., & Armentano, V. A. (2020). Logic-based Benders decomposition
        for the heterogeneous fixed fleet VRP with TW.
        Computers & Industrial Engineering, 148, 106652.

Attributes:
    run_lbbd_stage: Main entry point for Stage 1.

Example:
    >>> from logic.src.policies.routing.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd import run_lbbd_stage
    >>> routes, _ = run_lbbd_stage(env, wastes, capacity, R, C)
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.policies.routing.matheuristics.exact_guided_heuristic.route_pool import RoutePool, VRPPRoute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _route_cost_from_nodes(
    nodes: List[int],
    dist: np.ndarray,
    C: float,
) -> float:
    """Calculate the cost of a route from a list of nodes.

    Args:
        nodes: List of local indices to route.
        dist: Distance matrix.
        C: Cost per unit distance.

    Returns:
        Cost of the route.
    """
    path = [0] + nodes + [0]
    return C * sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))


def _route_revenue_from_nodes(
    nodes: List[int],
    wastes: Dict[int, float],
    R: float,
) -> float:
    """Calculate the revenue of a route from a list of nodes.

    Args:
        nodes: List of local indices to route.
        wastes: {local node → fill_level}.
        R: Revenue per unit waste.

    Returns:
        Revenue of the route.
    """
    return R * sum(wastes.get(n, 0.0) for n in nodes)


def _make_vrpp_route(
    nodes: List[int],
    dist: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    source: str,
) -> VRPPRoute:
    """Create a VRPPRoute from a list of nodes.

    Args:
        nodes: List of local indices to route.
        dist: Distance matrix.
        wastes: {local node → fill_level}.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        source: Source of the route.

    Returns:
        VRPPRoute.
    """
    cost = _route_cost_from_nodes(nodes, dist, C)
    revenue = _route_revenue_from_nodes(nodes, wastes, R)
    load = sum(wastes.get(n, 0.0) for n in nodes)
    return VRPPRoute(
        nodes=list(nodes),
        profit=revenue - cost,
        revenue=revenue,
        cost=cost,
        load=load,
        source=source,
    )


# ---------------------------------------------------------------------------
# Routing sub-problem oracles
# ---------------------------------------------------------------------------


def _solve_sub_greedy(
    selected: Set[int],
    dist: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory: Set[int],
) -> Tuple[List[List[int]], float]:
    """Greedy nearest-neighbour sub-solver (always fast, used for warm-start).

    Args:
        selected: Set of local indices to route.
        dist: Distance matrix.
        wastes: {local node → fill_level}.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        mandatory: Mandatory nodes.

    Returns:
        Tuple of routes and profit.
    """
    remaining = list(selected)
    routes: List[List[int]] = []
    while remaining:
        route: List[int] = []
        load = 0.0
        curr = 0
        while remaining:
            cands = [n for n in remaining if load + wastes.get(n, 0.0) <= capacity + 1e-6]
            if not cands:
                break
            nxt = min(cands, key=lambda n: dist[curr][n])
            route.append(nxt)
            load += wastes.get(nxt, 0.0)
            remaining.remove(nxt)
            curr = nxt
        if route:
            routes.append(route)
    sub_cost = sum(_route_cost_from_nodes(r, dist, C) for r in routes)
    sub_rev = sum(_route_revenue_from_nodes(r, wastes, R) for r in routes)
    return routes, sub_rev - sub_cost


def _solve_sub_alns(
    selected: Set[int],
    dist: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory: Set[int],
    time_limit: float,
    seed: Optional[int],
) -> Tuple[List[List[int]], float]:
    """ALNS sub-solver — reuses the project's ALNSSolver.

    Args:
        selected: Set of local indices to route.
        dist: Distance matrix.
        wastes: {local node → fill_level}.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        mandatory: Mandatory nodes.
        time_limit: Time limit for the sub-solver.
        seed: Random seed.

    Returns:
        Tuple of routes and profit.
    """
    try:
        from logic.src.configs.policies.alns import ALNSConfig
        from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import (
            ALNSSolver,
        )

        sub_wastes = {n: wastes.get(n, 0.0) for n in selected}
        cfg = ALNSConfig(
            time_limit=time_limit,
            max_iterations=max(200, int(500 * time_limit)),
            seed=seed,
            vrpp=True,
            profit_aware_operators=True,
        )
        solver = ALNSSolver(dist, sub_wastes, capacity, R, C, cfg, mandatory_nodes=sorted(mandatory & selected))
        routes, profit, _ = solver.solve()
        return routes, profit
    except Exception as exc:
        logger.debug("[LBBD-sub] ALNS fallback to greedy: %s", exc)
        return _solve_sub_greedy(selected, dist, wastes, capacity, R, C, mandatory)


def _solve_sub_bpc(
    selected: Set[int],
    dist: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory: Set[int],
    time_limit: float,
    seed: Optional[int],
    vehicle_limit: Optional[int],
    env: Any,
) -> Tuple[List[List[int]], float]:
    """BPC sub-solver — uses run_bpc for exact routing.

    Args:
        selected: Set of local indices to route.
        dist: Distance matrix.
        wastes: {local node → fill_level}.
        capacity: Vehicle capacity.
        R: Revenue per unit waste.
        C: Cost per unit distance.
        mandatory: Mandatory nodes.
        time_limit: Time limit for the sub-solver.
        seed: Random seed.
        vehicle_limit: Maximum number of vehicles.
        env: Gurobi environment.

    Returns:
        Tuple of routes and profit.
    """
    try:
        from logic.src.configs.policies.bpc import BPCConfig
        from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import (
            run_bpc,
        )

        sub_wastes = {n: wastes.get(n, 0.0) for n in selected}
        bpc_params = BPCConfig(time_limit=time_limit, seed=seed, vrpp=True)
        routes, profit = run_bpc(
            dist_matrix=dist,
            wastes=sub_wastes,
            capacity=capacity,
            R=R,
            C=C,
            params=bpc_params,
            mandatory_indices=mandatory & selected,
            vehicle_limit=vehicle_limit,
            env=env,
        )
        return routes, profit
    except Exception as exc:
        logger.debug("[LBBD-sub] BPC fallback to greedy: %s", exc)
        return _solve_sub_greedy(selected, dist, wastes, capacity, R, C, mandatory)


# ---------------------------------------------------------------------------
# LBBD master problem
# ---------------------------------------------------------------------------


class _LBBDMaster:
    """Gurobi MIP master that accumulates Benders cuts.

    Decision variables:
        y_i ∈ {0,1}  — whether customer i is visited.
        θ  ≥ 0       — routing-cost proxy (receives Benders cuts).

    Objective: max  Σ R·w_i·y_i  −  θ

    Attributes:
        n_nodes: Number of nodes.
        wastes: {local node → fill_level}.
        R: Revenue per unit waste.
        mandatory: Local indices of mandatory customers.
        env: Gurobi environment.
        seed: Random seed.
        mdl: Gurobi model.
        y: Decision variables for visit.
        theta: Routing cost proxy.
        _big_m: Big M constant.
        _cut_count: Number of cuts added.
    """

    def __init__(
        self,
        n_nodes: int,
        wastes: Dict[int, float],
        R: float,
        mandatory: Set[int],
        min_cover_ratio: float,
        vehicle_limit: Optional[int],
        env: Any,
        seed: int,
    ) -> None:
        """
        Args:
            n_nodes: Number of nodes.
            wastes: {local node → fill_level}.
            R: Revenue per unit waste.
            mandatory: Local indices of mandatory customers.
            min_cover_ratio: Minimum coverage ratio.
            vehicle_limit: Maximum number of vehicles.
            env: Gurobi environment.
            seed: Random seed.

        """
        self.n_nodes = n_nodes
        self.wastes = wastes
        self.R = R
        self.mandatory = mandatory
        self.env = env

        self.mdl = gp.Model("LBBD_Master", env=env) if env else gp.Model("LBBD_Master")
        self.mdl.Params.LogToConsole = 0
        self.mdl.Params.OutputFlag = 0
        self.mdl.Params.Seed = seed

        # y_i: visit binary
        self.y = self.mdl.addVars(range(1, n_nodes + 1), vtype=GRB.BINARY, name="y")
        # θ: routing cost proxy
        self.theta = self.mdl.addVar(lb=0.0, name="theta")

        # Mandatory coverage
        for i in mandatory:
            self.mdl.addConstr(self.y[i] == 1, name=f"mand_{i}")

        # Minimum coverage ratio
        if min_cover_ratio < 1.0:
            self.mdl.addConstr(
                quicksum(self.y[i] for i in range(1, n_nodes + 1)) >= math.ceil(min_cover_ratio * n_nodes),
                name="min_cover",
            )

        revenue_expr = quicksum(R * wastes.get(i, 0.0) * self.y[i] for i in range(1, n_nodes + 1))
        self.mdl.setObjective(revenue_expr - self.theta, GRB.MAXIMIZE)
        self.mdl.update()

        self._big_m = R * sum(wastes.values()) + 1.0
        self._cut_count = 0

    def solve(self, time_limit: float) -> Optional[Set[int]]:
        """Solve master and return selected node set, or None if infeasible.

        Args:
            time_limit: Time limit.

        Returns:
            Set of selected nodes.
        """
        self.mdl.Params.TimeLimit = max(1.0, time_limit)
        self.mdl.optimize()
        if self.mdl.SolCount == 0:
            return None
        return {i for i in range(1, self.n_nodes + 1) if self.y[i].X > 0.5}

    def get_lp_bound(self) -> float:
        """LP relaxation upper bound after last solve.

        Args:
            None

        Returns:
            LP relaxation upper bound.
        """
        try:
            return float(self.mdl.ObjBound)
        except Exception:
            return float("inf")

    def add_nogood_cut(self, Y: Set[int]) -> None:
        """Forbid the exact selection Y.

        Args:
            Y: Set of nodes.

        """
        expr = quicksum(self.y[i] for i in Y)
        self.mdl.addConstr(expr <= len(Y) - 1, name=f"nogood_{self._cut_count}")
        self._cut_count += 1
        self.mdl.update()

    def add_optimality_cut(self, Y: Set[int], sub_cost: float) -> None:
        """Add: θ ≥ sub_cost − M · (|Y| − Σ_{i∈Y} y_i).

        Args:
            Y: Set of nodes.
            sub_cost: Subproblem cost.

        """
        lhs = quicksum(self.y[i] for i in Y)
        self.mdl.addConstr(
            self.theta >= sub_cost - self._big_m * (len(Y) - lhs),
            name=f"opt_{self._cut_count}",
        )
        self._cut_count += 1
        self.mdl.update()

    def add_pareto_cut(self, Y: Set[int], sub_cost: float, primal_bound: float) -> None:
        """Magnanti-Wang Pareto-optimal strengthening of the optimality cut.

        Args:
            Y: Set of nodes.
            sub_cost: Subproblem cost.
            primal_bound: Primal bound.

        """
        gap = sub_cost - primal_bound
        if gap <= 1e-8:
            return
        lhs = quicksum(self.y[i] for i in Y)
        self.mdl.addConstr(
            self.theta >= gap * lhs - gap * len(Y) + primal_bound,
            name=f"pareto_{self._cut_count}",
        )
        self._cut_count += 1
        self.mdl.update()

    def add_combinatorial_cut(self, Y: Set[int], sub_cost: float) -> None:
        """Coverage-strengthening cut: every subset of Y needs the same routing.

        Args:
            Y: Set of nodes.
            sub_cost: Subproblem cost.

        """
        # For any T ⊆ Y, cost(T) ≥ cost(Y) − Σ_{i∉T, i∈Y} R·w_i
        # Approximated as: θ ≥ sub_cost − Σ_{i∈Y} R·w_i · (1 − y_i)
        penalty = quicksum(self.R * self.wastes.get(i, 0.0) * (1 - self.y[i]) for i in Y)
        self.mdl.addConstr(
            self.theta >= sub_cost - penalty,
            name=f"comb_{self._cut_count}",
        )
        self._cut_count += 1
        self.mdl.update()

    @property
    def n_cuts(self) -> int:
        """Return the number of cuts added to the master problem.

        Returns:
            Number of cuts added.
        """
        return self._cut_count


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_lbbd_stage(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory: Set[int],
    n_vehicles: int,
    time_limit: float,
    max_iterations: int,
    sub_solver: str,
    cut_families: List[str],
    pareto_eps: float,
    min_cover_ratio: float,
    master_time_frac: float,
    sub_time_frac: float,
    pool: Optional[RoutePool],
    seed: int,
    env: Any,
    incumbent: float = 0.0,
) -> Tuple[List[VRPPRoute], float, float]:
    """Execute LBBD and return routes, best profit, and LP upper bound.

    Args:
        dist_matrix:      Full distance matrix (depot at index 0).
        wastes:           {local node → fill_level}.
        capacity:         Vehicle capacity Q.
        R:                Revenue per unit waste.
        C:                Cost per unit distance.
        mandatory:        Local indices of mandatory customers.
        n_vehicles:       Fleet size K.
        time_limit:       Total LBBD stage budget (seconds).
        max_iterations:   Maximum outer Benders iterations.
        sub_solver:       'greedy' | 'alns' | 'bpc'.
        cut_families:     Active cut families (list of strings).
        pareto_eps:       Tolerance for Pareto-optimal cut.
        min_cover_ratio:  Min fraction of nodes the master must cover.
        master_time_frac: Fraction of per-iteration budget for master.
        sub_time_frac:    Fraction of per-iteration budget for sub.
        pool:             Shared RoutePool.
        seed:             RNG seed.
        env:              Shared Gurobi environment.
        incumbent:        Best profit known before LBBD starts.

    Returns:
        (best_routes, best_profit, lp_upper_bound)
    """
    n_nodes = len(dist_matrix) - 1
    t_start = time.perf_counter()
    per_iter = max(2.0, time_limit / max(1, max_iterations))

    # Resolve sub-problem oracle
    vehicle_lim = n_vehicles if n_vehicles > 0 else None
    _sub_dispatch: Dict[str, Callable] = {
        "greedy": lambda Y, tl: _solve_sub_greedy(Y, dist_matrix, wastes, capacity, R, C, mandatory),
        "alns": lambda Y, tl: _solve_sub_alns(Y, dist_matrix, wastes, capacity, R, C, mandatory, tl, seed),
        "bpc": lambda Y, tl: _solve_sub_bpc(
            Y, dist_matrix, wastes, capacity, R, C, mandatory, tl, seed, vehicle_lim, env
        ),
    }
    sub_fn = _sub_dispatch.get(sub_solver, _sub_dispatch["greedy"])

    master = _LBBDMaster(
        n_nodes=n_nodes,
        wastes=wastes,
        R=R,
        mandatory=mandatory,
        min_cover_ratio=min_cover_ratio,
        vehicle_limit=vehicle_lim,
        env=env,
        seed=seed,
    )

    best_routes: List[VRPPRoute] = []
    best_profit: float = incumbent
    lp_ub: float = float("inf")
    visited: Set[frozenset] = set()

    for iteration in range(max_iterations):
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        remaining = time_limit - elapsed
        master_tl = min(per_iter * master_time_frac, remaining * 0.3)
        sub_tl = min(per_iter * sub_time_frac, remaining * 0.7)

        # ── Master solve ────────────────────────────────────────────────
        Y = master.solve(master_tl)
        if Y is None:
            logger.info("[LBBD] Master infeasible at iter %d — stopping", iteration)
            break

        lp_ub = min(lp_ub, master.get_lp_bound())

        key = frozenset(Y)
        if key in visited:
            # Pure no-good loop — add another no-good to escape and continue
            master.add_nogood_cut(Y)
            continue
        visited.add(key)

        # ── Sub-problem solve ───────────────────────────────────────────
        sub_routes_raw, sub_profit = sub_fn(Y, sub_tl)

        # Record sub-problem routes into pool
        for r in sub_routes_raw:
            inner = [nd for nd in r if nd != 0]
            if inner:
                vr = _make_vrpp_route(inner, dist_matrix, wastes, R, C, f"lbbd_{sub_solver}")
                if pool is not None:
                    pool.add(vr)
                if sub_profit > best_profit + 1e-9:
                    best_routes = [vr]
                    best_profit = sub_profit

        # Routing cost for cut generation
        sub_cost = sum(_route_cost_from_nodes([nd for nd in r if nd != 0], dist_matrix, C) for r in sub_routes_raw)

        # ── Cut generation ──────────────────────────────────────────────
        if "nogood" in cut_families:
            master.add_nogood_cut(Y)

        if "optimality" in cut_families and sub_cost < float("inf"):
            master.add_optimality_cut(Y, sub_cost)

        if "pareto" in cut_families and sub_cost < float("inf"):
            master.add_pareto_cut(Y, sub_cost, best_profit)

        if "combinatorial" in cut_families and sub_cost < float("inf"):
            master.add_combinatorial_cut(Y, sub_cost)

        logger.debug(
            "[LBBD] iter=%d  |Y|=%d  sub_profit=%.4f  best=%.4f  cuts=%d",
            iteration,
            len(Y),
            sub_profit,
            best_profit,
            master.n_cuts,
        )

        # Convergence: LP bound ≈ primal
        if lp_ub - best_profit <= pareto_eps * max(1.0, abs(best_profit)):
            logger.info("[LBBD] Converged at iter %d (gap=%.6f)", iteration, lp_ub - best_profit)
            break

    logger.info(
        "[LBBD] done  iters=%d  profit=%.4f  lp_ub=%.4f  cuts=%d  pool=%s",
        iteration + 1,
        best_profit,
        lp_ub,
        master.n_cuts,
        len(pool) if pool else "n/a",
    )
    return best_routes, best_profit, lp_ub
