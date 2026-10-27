"""Set-Cover Route Improver (Pool-Restricted Exact).

This module builds a route pool from the input tour plus perturbations and
solves the set-cover IP exactly via Gurobi. Unlike Set Partitioning, Set Cover
allows nodes to be visited in multiple selected routes if it yields a superior
packing. A post-processing step removes overlaps to restore a valid partition.

Attributes:
    SetCoverRouteImprover: Route improvement class using Set Covering.

Example:
    >>> improver = SetCoverRouteImprover()
    >>> best_tour, metrics = improver.process(tour, distance_matrix=dm)
"""

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, cast

import numpy as np

from logic.src.enums import PolicyTag
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.intensification_fixing import dp_route_reopt
from logic.src.policies.helpers.operators.search_heuristics.large_neighborhood_search import apply_lns
from logic.src.utils.registry import ALL

from .base import RouteImproverRegistry
from .common.helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    split_tour,
    to_numpy,
    tour_distance,
    upgrade_repair_op_to_profit,
)

logger = logging.getLogger(__name__)

try:
    import gurobipy as gp

    _HAS_GUROBI = True
    try:
        _test = gp.Model()
        _test.dispose()
    except gp.GurobiError as e:
        _HAS_GUROBI = False
        logger.warning(
            "set_cover: Gurobi license check failed (%s); route improver will no-op.",
            e,
        )
except ImportError:
    _HAS_GUROBI = False
    logger.warning("set_cover: gurobipy not installed; route improver will no-op.")


def _canonical(route: List[int]) -> Tuple[int, ...]:
    """Canonical form for symmetric-distance deduplication.

    Args:
        route: The route to canonicalize.

    Returns:
        Tuple[int, ...]: The canonical representation of the route.
    """
    fwd = tuple(route)
    rev = tuple(reversed(route))
    return min(fwd, rev)


@ALL.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.MATHEURISTIC,
    PolicyTag.MATH_PROGRAMMING,
)
@RouteImproverRegistry.register("set_cover")
class SetCoverRouteImprover(IRouteImprovement):
    """Set-Cover route improver.

    Attributes:
        config (Dict[str, Any]): Configuration parameters.

    Example:
        >>> improver = SetCoverRouteImprover()
        >>> tour, metrics = improver.process(tour, distance_matrix=dm)
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:  # noqa: C901
        """Apply Set Cover intensification to the tour.

        Args:
            tour: Initial tour sequence (list of bin IDs including depot 0s).
            kwargs: Search context, including:
                - distance_matrix (np.ndarray): The distance matrix.
                - sp_n_perturbations (int): Number of LNS perturbations for pool.
                - sp_include_dp (bool): Whether to include DP-optimized variants.
                - dp_max_nodes (int): Max nodes for DP re-optimization.
                - ruin_fraction (float): LNS ruin fraction.
                - sp_time_limit (float): Gurobi time limit.
                - wastes (Dict[int, float]): Bin waste demands.
                - capacity (float): Vehicle capacity.
                - cost_per_km (float): Distance cost.
                - revenue_kg (float): Waste revenue.
                - seed (int): Random seed.

        Returns:
            Tuple[List[int], ImprovementMetrics]: Refined tour and performance metrics.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "SetCoverRouteImprover"}

        if not _HAS_GUROBI:
            return tour, {"algorithm": "SetCoverRouteImprover"}

        # Pool-construction parameters
        n_perturbations = cast(int, kwargs.get("sp_n_perturbations", self.config.get("sp_n_perturbations", 20)))
        include_dp_variants = kwargs.get("sp_include_dp", self.config.get("sp_include_dp", True))
        dp_max_nodes = cast(int, kwargs.get("dp_max_nodes", self.config.get("dp_max_nodes", 20)))
        ruin_fraction = cast(float, kwargs.get("ruin_fraction", self.config.get("ruin_fraction", 0.2)))
        destroy_op = cast(str, kwargs.get("destroy_op", self.config.get("destroy_op", "random")))
        repair_op_raw = cast(str, kwargs.get("repair_op", self.config.get("repair_op", "greedy")))

        # Solver parameters
        time_limit = cast(float, kwargs.get("sp_time_limit", self.config.get("sp_time_limit", 60.0)))
        seed = cast(int, kwargs.get("seed", self.config.get("seed", 42)))

        # Problem parameters
        wastes = cast(Dict[int, float], kwargs.get("wastes", self.config.get("wastes", {})))
        capacity = cast(float, kwargs.get("capacity", self.config.get("capacity", float("inf"))))
        cost_per_km = cast(float, kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0)))
        revenue_kg = cast(float, kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0)))
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config)

        repair_op = upgrade_repair_op_to_profit(repair_op_raw, revenue_kg, cost_per_km)

        dm = to_numpy(distance_matrix)

        try:
            input_routes = split_tour(tour)
            if not input_routes:
                return tour, {"algorithm": "SetCoverRouteImprover"}

            # --- Pool construction ---
            seen: Set[Tuple[int, ...]] = set()
            pool: List[List[int]] = []

            def add(route: List[int]) -> None:
                """Adds a candidate route to the pool if it is unique.

                Args:
                    route: The sequence of node IDs to potentially add.

                """
                if not route:
                    return
                key = _canonical(route)
                if key not in seen:
                    seen.add(key)
                    pool.append(list(route))

            for r in input_routes:
                add(r)

            base_rng = random.Random(seed)
            for i in range(n_perturbations):
                perturb_rng = random.Random(base_rng.randint(0, 2**31 - 1))
                try:
                    perturbed = apply_lns(
                        routes=input_routes,
                        dist_matrix=dm,
                        wastes=wastes,
                        capacity=capacity,
                        R=revenue_kg,
                        C=cost_per_km,
                        rng=perturb_rng,
                        ruin_fraction=ruin_fraction,
                        destroy_op=destroy_op,
                        repair_op=repair_op,
                        mandatory_nodes=mandatory_nodes,
                    )
                    for r in perturbed:
                        add(r)
                except Exception as e:
                    logger.debug("set_cover: perturbation %d failed: %s", i, e)
                    continue

            if include_dp_variants:
                try:
                    dp_routes = dp_route_reopt(
                        routes=input_routes,
                        dist_matrix=dm,
                        wastes=wastes,
                        capacity=capacity,
                        max_nodes=dp_max_nodes,
                    )
                    for r in dp_routes:
                        add(r)
                except Exception as e:
                    logger.debug("set_cover: dp_route_reopt failed: %s", e)

            if mandatory_nodes:
                covered = {n for r in pool for n in r}
                for m_node in mandatory_nodes:
                    if m_node not in covered:
                        add([0, m_node, 0])

            if not pool:
                return tour, {"algorithm": "SetCoverRouteImprover"}

            # --- Target Nodes ---
            target_nodes = (
                set(mandatory_nodes) if mandatory_nodes else set(n for r in input_routes for n in r if n != 0)
            )

            # --- Solve Set Cover IP ---
            selected_routes = self._solve_set_cover_ip(
                pool=pool,
                dm=dm,
                wastes=wastes,
                cost_per_km=cost_per_km,
                revenue_kg=revenue_kg,
                target_nodes=target_nodes,
                time_limit=time_limit,
            )

            if not selected_routes:
                return tour, {"algorithm": "SetCoverRouteImprover"}

            # --- Post-processing: Remove Overlaps ---
            refined_routes = self._deduplicate_nodes(selected_routes, dm)

            # Acceptance gate: verify improvement
            polished_cost = tour_distance(refined_routes, dm)
            input_cost = tour_distance(input_routes, dm)
            if polished_cost > input_cost + 1e-6:
                return tour, {"algorithm": "SetCoverRouteImprover"}

            return assemble_tour(refined_routes), {"algorithm": "SetCoverRouteImprover"}

        except Exception as e:
            logger.debug("set_cover final fallback invoked: %s", e)
            return tour, {"algorithm": "SetCoverRouteImprover"}

    def _solve_set_cover_ip(
        self,
        pool: List[List[int]],
        dm: np.ndarray,
        wastes: dict,
        cost_per_km: float,
        revenue_kg: float,
        target_nodes: Set[int],
        time_limit: float,
    ) -> List[List[int]]:
        """Builds and solves the Set Cover formulation using Gurobi.

        Args:
            pool: Candidate routes.
            dm: Distance matrix.
            wastes: Bin waste demands.
            cost_per_km: Distance cost.
            revenue_kg: Waste revenue.
            target_nodes: Nodes that must be covered.
            time_limit: Gurobi time limit.

        Returns:
            List[List[int]]: Selected subset of routes.
        """
        model = gp.Model("SetCover")
        model.setParam("TimeLimit", time_limit)
        model.setParam("OutputFlag", 0)
        model.setParam("LogToConsole", 0)

        x = {}
        for idx, route in enumerate(pool):
            dist = tour_distance([route], dm)
            # Default to pure distance minimization if C and R are both zero
            cost = dist * cost_per_km if cost_per_km > 0 else dist
            profit = sum(wastes.get(n, 0.0) for n in set(route) if n != 0) * revenue_kg

            # Objective: Minimize (Cost - Profit)
            x[idx] = model.addVar(vtype=gp.GRB.BINARY, obj=(cost - profit), name=f"x_{idx}")

        # Set Cover Constraint: sum(x_r) >= 1 for all r containing node
        for node in target_nodes:
            covering_routes = [idx for idx, r in enumerate(pool) if node in r]
            if covering_routes:
                model.addConstr(gp.quicksum(x[idx] for idx in covering_routes) >= 1, name=f"cov_{node}")

        model.optimize()

        if model.Status in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT) and model.SolCount > 0:
            return [list(pool[idx]) for idx in x if x[idx].X > 0.5]

        return []

    def _deduplicate_nodes(self, routes: List[List[int]], dm: np.ndarray) -> List[List[int]]:
        """Removes duplicate nodes from overlapping routes.

        Identifies nodes visited more than once and drops them from the route
        where their removal yields the greatest distance reduction.

        Args:
            routes: Selected routes with overlaps.
            dm: Distance matrix.

        Returns:
            List[List[int]]: Routes with overlaps resolved.
        """
        # Count occurrences of all non-depot nodes
        counts: Dict[int, int] = defaultdict(int)
        for r in routes:
            for n in r:
                if n != 0:
                    counts[n] += 1

        for node, count in counts.items():
            while count > 1:
                best_savings = -float("inf")
                best_route_idx = -1
                best_node_idx = -1

                # Evaluate removal savings across all routes containing the node
                for i, r in enumerate(routes):
                    for j in range(1, len(r) - 1):
                        if r[j] == node:
                            prev_n = r[j - 1]
                            next_n = r[j + 1]
                            # Triangle inequality savings: D(prev, node) + D(node, next) - D(prev, next)
                            savings = dm[prev_n, node] + dm[node, next_n] - dm[prev_n, next_n]

                            if savings > best_savings:
                                best_savings = savings
                                best_route_idx = i
                                best_node_idx = j

                if best_route_idx != -1:
                    routes[best_route_idx].pop(best_node_idx)
                    count -= 1
                else:
                    break  # Fallback if structural integrity fails

        # Clean up any routes that became empty [0, 0] during deduplication
        return [r for r in routes if len(r) > 2]
