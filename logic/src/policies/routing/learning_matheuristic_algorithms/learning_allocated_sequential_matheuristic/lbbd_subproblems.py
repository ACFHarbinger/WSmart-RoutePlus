"""Route-cost helpers and the greedy, ALNS and branch-and-price-and-cut subproblem oracles used by the LBBD stage."""

from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np

from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.matheuristics.exact_guided_heuristic.route_pool import VRPPRoute

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
        from logic.src.configs.policies.routing.meta_heuristics.alns import ALNSConfig
        from logic.src.policies.routing.meta_heuristics.adaptive_large_neighborhood_search.alns import (
            ALNSSolver,
        )

        sub_wastes = {n: wastes.get(n, 0.0) for n in selected}
        cfg = ALNSConfig(
            time_limit=time_limit,
            max_iterations=max(200, int(500 * time_limit)),
            seed=seed,
            profit_aware_operators=True,
        )
        solver = ALNSSolver(
            build_subproblem(
                dist_matrix=dist,
                wastes=sub_wastes,
                capacity=capacity,
                revenue=R,
                cost_unit=C,
                mandatory_nodes=sorted(mandatory & selected),
            ),
            cfg,
        )
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
        from logic.src.configs.policies.routing.exact_and_decomposition_solvers.bpc import BPCConfig
        from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import (
            run_bpc,
        )

        sub_wastes = {n: wastes.get(n, 0.0) for n in selected}
        bpc_params = BPCConfig(time_limit=time_limit, seed=seed)
        routes, profit, _cost = run_bpc(
            build_subproblem(
                dist_matrix=dist,
                wastes=sub_wastes,
                capacity=capacity,
                revenue=R,
                cost_unit=C,
                mandatory_nodes=sorted(mandatory & selected),
                n_vehicles=vehicle_limit,
                env=env,
            ),
            bpc_params,
        )
        return routes, profit
    except Exception as exc:
        logger.debug("[LBBD-sub] BPC fallback to greedy: %s", exc)
        return _solve_sub_greedy(selected, dist, wastes, capacity, R, C, mandatory)
