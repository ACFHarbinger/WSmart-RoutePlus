"""Greedy warm start of the restricted master's column pool.

BPC and MS-BPC-SP seed the master with greedy-insertion routes before column
generation: routes over the mandatory bins, plus (profit-aware variant only)
routes extended with profitable optional bins, so the first LP already prices
optional bins. With the warm start off the master starts empty and column
generation (Phase I Farkas pricing when mandatory bins are uncovered) builds
the whole pool.

Attributes:
    greedy_warm_start_columns: Build the initial routes and columns.

Example:
    >>> routes, columns = greedy_warm_start_columns(pricer, [1, 2], [3], dist, wastes, 100.0, 1.0, 1.0)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from logic.src.policies.helpers.operators.recreate_repair.greedy import greedy_insertion, greedy_profit_insertion

logger = logging.getLogger(__name__)


def greedy_warm_start_columns(
    pricing_solver: Any,
    mandatory: Sequence[int],
    optional: Sequence[int],
    dist: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    *,
    enabled: bool = True,
    profit_aware: bool = True,
) -> Tuple[List[List[int]], List[Any]]:
    """Build the master's initial routes and columns by greedy insertion.

    Args:
        pricing_solver: The RCSPP solver; its ``_compute_route_details`` turns a
            node list into a ``Route`` column.
        mandatory: Local ids of the mandatory bins.
        optional: Local ids of the optional bins.
        dist: Distance matrix of the (reduced) instance, depot at 0.
        wastes: Local bin id -> waste.
        capacity: Vehicle capacity.
        R: Revenue per kg.
        C: Cost per km.
        enabled: False skips the warm start: no routes, no columns.
        profit_aware: Use profit-aware insertion and seed optional-bin routes.

    Returns:
        ``(routes, columns)``: the mandatory-bin greedy routes (node lists, the
        engines' fallback plan) and the columns for ``build_model``.
    """
    if not enabled:
        logger.info("[Warm-start] Greedy insertion off: the master starts with an empty column pool.")
        return [], []

    mandatory = list(mandatory)
    if profit_aware:
        routes = greedy_profit_insertion(
            [], mandatory, dist, wastes, capacity, R=R, C=C, mandatory_nodes=mandatory, expand_pool=False
        )
    else:
        routes = greedy_insertion([], mandatory, dist, wastes, capacity, mandatory_nodes=mandatory, expand_pool=False)
    columns = [pricing_solver._compute_route_details(r) for r in routes]

    # Seed with greedy optional-node routes so LP duals for optional nodes
    # are non-zero from the start, guiding the RCSPP toward optional nodes.
    if optional and profit_aware:
        try:
            seeded = greedy_profit_insertion(
                # Copies: the operator inserts into these lists in place, and the
                # mandatory columns built above hold the originals as Route.nodes.
                [list(r) for r in routes],
                list(optional),
                dist,
                wastes,
                capacity,
                R=R,
                C=C,
                mandatory_nodes=mandatory,
                expand_pool=True,
            )
            known = {tuple(r) for r in routes}
            for r_nodes in seeded:
                if tuple(r_nodes) not in known:
                    route_obj = pricing_solver._compute_route_details(r_nodes)
                    if route_obj.profit > 0:
                        columns.append(route_obj)
            logger.info(
                f"[Warm-start] Added {len(seeded) - len(routes)} optional-node seed routes to initial column pool."
            )
        except Exception as _e:
            logger.debug(f"[Warm-start] Optional seeding failed: {_e}")
    return routes, columns
