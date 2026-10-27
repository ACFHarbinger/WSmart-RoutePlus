"""Fix-and-Optimize Route Improver.

Delegates to operators.intensification.fix_and_optimize (or its profit
variant when revenue/cost are configured). Selects the worst-quality
routes, solves them exactly via a Gurobi sub-MIP, and recombines with
the unchanged fixed routes.

Attributes:
    FixAndOptimizeRouteImprover: Main class for sub-MIP intensification.

Example:
    >>> improver = FixAndOptimizeRouteImprover(config=cfg)
    >>> refined_tour, metrics = improver.process(tour, fo_n_free=3)
"""

import logging
from typing import Any, Dict, List, Tuple, cast

from logic.src.enums import PolicyTag
from logic.src.interfaces.context.search_context import ImprovementMetrics
from logic.src.interfaces.route_improvement import IRouteImprovement
from logic.src.policies.helpers.operators.intensification_fixing import (
    fix_and_optimize,
    fix_and_optimize_profit,
)
from logic.src.utils.registry import ALL

from .base import RouteImproverRegistry
from .common.helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    split_tour,
    to_numpy,
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
            "fix_and_optimize: Gurobi license check failed (%s); route improver will no-op.",
            e,
        )
except ImportError:
    _HAS_GUROBI = False
    logger.warning("fix_and_optimize: gurobipy not installed; route improver will no-op.")


@ALL.register(
    PolicyTag.IMPROVEMENT,
    PolicyTag.MATHEURISTIC,
    PolicyTag.MATH_PROGRAMMING,
)
@RouteImproverRegistry.register("fix_and_optimize")
class FixAndOptimizeRouteImprover(IRouteImprovement):
    """Fix-and-Optimize sub-MIP route improver.

    Ranks routes by quality (distance-per-load for CVRP, profit for VRPP),
    designates the `fo_free_fraction` worst routes as "free", and solves
    their customers as a small Gurobi sub-MIP. The optimal recombination
    replaces the free routes; all other routes pass through unchanged.

    Attributes:
        config (Dict[str, Any]): Internal configuration state.

    Example:
        >>> improver = FixAndOptimizeRouteImprover(config=cfg)
        >>> refined_tour, metrics = improver.process(tour, fo_free_fraction=0.5)
    """

    def process(self, tour: List[int], **kwargs: Any) -> Tuple[List[int], ImprovementMetrics]:
        """Apply sub-MIP reoptimization to the tour.

        Args:
            tour: Initial tour sequence.
            kwargs: Context containing:
                distance_matrix: Distance lookup.
                fo_n_free: Absolute number of routes to free (overrides fraction).
                fo_free_fraction: Fraction of routes to free (default 0.30).
                fo_time_limit: Seconds for Gurobi solve (default 30.0).
                seed: Random seed.
                wastes: Bin mass dictionary.
                capacity: Maximum vehicle capacity.
                revenue_kg: Waste profit coefficient.
                cost_per_km: Distance cost coefficient.
                mandatory_nodes: List of required visitor IDs.

        Returns:
            Tuple[List[int], ImprovementMetrics]: Refined tour and metrics.
        """
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour, {"algorithm": "FixAndOptimizeRouteImprover"}

        if not _HAS_GUROBI:
            return tour, {"algorithm": "FixAndOptimizeRouteImprover"}

        # Sub-MIP parameters
        n_free = kwargs.get("fo_n_free", self.config.get("fo_n_free"))  # None -> use fraction
        free_fraction = cast(float, kwargs.get("fo_free_fraction", self.config.get("fo_free_fraction", 0.30)))
        time_limit = cast(float, kwargs.get("fo_time_limit", self.config.get("fo_time_limit", 30.0)))
        seed = cast(int, kwargs.get("seed", self.config.get("seed", 42)))

        # Problem parameters
        wastes = cast(Dict[int, float], kwargs.get("wastes", self.config.get("wastes", {})))
        capacity = cast(float, kwargs.get("capacity", self.config.get("capacity", float("inf"))))
        cost_per_km = cast(float, kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0)))
        revenue_kg = cast(float, kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0)))
        mandatory_nodes = resolve_mandatory_nodes(kwargs, self.config)

        dm = to_numpy(distance_matrix)

        try:
            routes = split_tour(tour)
            if not routes:
                return tour, {"algorithm": "FixAndOptimizeRouteImprover"}

            if revenue_kg > 0 or cost_per_km > 0:
                refined = fix_and_optimize_profit(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    n_free=n_free,
                    free_fraction=free_fraction,
                    time_limit=time_limit,
                    seed=seed,
                    mandatory_nodes=mandatory_nodes,
                )
            else:
                refined = fix_and_optimize(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    n_free=n_free,
                    free_fraction=free_fraction,
                    time_limit=time_limit,
                    seed=seed,
                )

            return assemble_tour(refined), {"algorithm": "FixAndOptimizeRouteImprover"}

        except Exception:
            return tour, {"algorithm": "FixAndOptimizeRouteImprover"}
