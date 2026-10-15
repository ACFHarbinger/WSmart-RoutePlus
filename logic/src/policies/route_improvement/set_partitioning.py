"""Set-Partitioning Route Improver (Pool-Restricted Exact).

This module builds a route pool from the input tour plus perturbations and
solves the set-partitioning IP exactly via Gurobi. The result is optimal
within the provided pool. Unlike Set Covering, each node must appear exactly
once in the selected routes.

Attributes:
    SetPartitioningRouteImprover: Route improvement class using Set Partitioning.

Example:
    >>> improver = SetPartitioningRouteImprover()
    >>> best_tour, metrics = improver.process(tour, distance_matrix=dm)
"""

import logging
import random
from typing import Any, List, Set, Tuple

from logic.src.enums import GlobalRegistry, PolicyTag
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.intensification_fixing import (
    dp_route_reopt,
    set_partitioning_polish,
    set_partitioning_polish_profit,
)
from logic.src.policies.helpers.operators.search_heuristics.large_neighborhood_search import apply_lns

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
    # Quick license ping
    try:
        _test = gp.Model()
        _test.dispose()
    except gp.GurobiError as e:
        _HAS_GUROBI = False
        logger.warning(
            "set_partitioning: Gurobi license check failed (%s); route improver will no-op.",
            e,
        )
except ImportError:
    _HAS_GUROBI = False
    logger.warning("set_partitioning: gurobipy not installed; route improver will no-op.")


def _canonical(route: List[int]) -> Tuple[int, ...]:
    """Canonical form for symmetric-distance deduplication.

    Args:
        route: The route sequence to canonicalize.

    Returns:
        Tuple[int, ...]: The canonical representation of the route.
    """
    fwd = tuple(route)
    rev = tuple(reversed(route))
    return min(fwd, rev)


@GlobalRegistry.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.MATHEURISTIC,
    PolicyTag.MATH_PROGRAMMING,
)
@RouteImproverRegistry.register("set_partitioning")
class SetPartitioningRouteImprover(IRouteImprovement):
    """Set-Partitioning route improver.

    Constructs a route pool by combining the input tour with multiple
    ruin-and-recreate perturbations and per-route DP optimisations, then
    solves the IP exactly with Gurobi.

    Attributes:
        config (Dict[str, Any]): Configuration parameters.

    Example:
        >>> improver = SetPartitioningRouteImprover()
        >>> tour, metrics = improver.process(tour, distance_matrix=dm)
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:  # noqa: C901
        """Apply Set Partitioning intensification to the tour.

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
            return tour, {"algorithm": "SetPartitioningRouteImprover"}

        if not _HAS_GUROBI:
            return tour, {"algorithm": "SetPartitioningRouteImprover"}

        # Pool-construction parameters
        n_perturbations = kwargs.get("sp_n_perturbations", self.config.get("sp_n_perturbations", 20))
        include_dp_variants = kwargs.get("sp_include_dp", self.config.get("sp_include_dp", True))
        dp_max_nodes = kwargs.get("dp_max_nodes", self.config.get("dp_max_nodes", 20))
        ruin_fraction = kwargs.get("ruin_fraction", self.config.get("ruin_fraction", 0.2))
        destroy_op = kwargs.get("destroy_op", self.config.get("destroy_op", "random"))
        repair_op_raw = kwargs.get("repair_op", self.config.get("repair_op", "greedy"))

        # Solver parameters
        time_limit = kwargs.get("sp_time_limit", self.config.get("sp_time_limit", 60.0))
        seed = kwargs.get("seed", self.config.get("seed", 42))

        # Problem parameters
        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        cost_per_km = kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0))
        revenue_kg = kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0))
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config)

        repair_op = upgrade_repair_op_to_profit(repair_op_raw, revenue_kg, cost_per_km)

        dm = to_numpy(distance_matrix)

        try:
            input_routes = split_tour(tour)
            if not input_routes:
                return tour, {"algorithm": "SetPartitioningRouteImprover"}

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

            # Source 1: input routes
            for r in input_routes:
                add(r)

            # Source 2: perturbations via apply_lns
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
                    logger.debug("set_partitioning: perturbation %d failed: %s", i, e)
                    continue  # one bad perturbation should not kill pool construction

            # Source 3: Held-Karp DP variants
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
                    logger.debug("set_partitioning: dp_route_reopt failed: %s", e)

            # Source 4: mandatory singletons
            if mandatory_nodes:
                covered = {n for r in pool for n in r}
                for m_node in mandatory_nodes:
                    if m_node not in covered:
                        add([m_node])

            if not pool:
                return tour, {"algorithm": "SetPartitioningRouteImprover"}

            # --- Solve set-partitioning IP ---
            if revenue_kg > 0 or cost_per_km > 0:
                refined = set_partitioning_polish_profit(
                    routes=input_routes,
                    route_pool=pool,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    mandatory_nodes=mandatory_nodes,
                    time_limit=time_limit,
                    seed=seed,
                )
            else:
                refined = set_partitioning_polish(
                    routes=input_routes,
                    route_pool=pool,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    time_limit=time_limit,
                    seed=seed,
                )

            # Acceptance gate: only return the polished tour if it's actually better.
            polished_cost = tour_distance(refined, dm)
            input_cost = tour_distance(input_routes, dm)
            if polished_cost > input_cost + 1e-6:
                return tour, {"algorithm": "SetPartitioningRouteImprover"}

            return assemble_tour(refined), {"algorithm": "SetPartitioningRouteImprover"}

        except Exception:
            return tour, {"algorithm": "SetPartitioningRouteImprover"}
