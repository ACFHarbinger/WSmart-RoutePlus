"""bpc options definitions relocated from bpc_tree."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BPCSearchOptions:
    """Flags that keep BPC and MS-BPC-SP certificates on their own paths.

    Attributes:
        strict_certification: Incomplete Farkas/pricing leaves the node
            uncertified (MS) instead of accepting a weak bound (BPC default).
        require_elementary: Integer check also requires elementary routes
            and a set-partition of visited nodes (MS).
        incomplete_node_ip: On node timeout, salvage an IP incumbent before
            breaking (MS). BPC just terminates the search.
        lr_requires_vehicle_limit: Skip Lagrangian pre-pruning when the
            fleet is unlimited (BPC). MS runs LR without that gate.
        scale_lr_by_vehicle_limit: Multiply the single-route LR bound by
            ``master.vehicle_limit`` (BPC). MS uses the raw bound.
    """

    strict_certification: bool = False
    require_elementary: bool = False
    incomplete_node_ip: bool = False
    lr_requires_vehicle_limit: bool = True
    scale_lr_by_vehicle_limit: bool = True


def total_travel_cost(routes: List[List[int]], dist_matrix: np.ndarray, C: float) -> float:
    """Monetary travel cost C × total route distance for a route set.

    Strip any leading/trailing depot index, wrap each route in a depot
    round trip, and accumulate the matrix distance.
    """
    raw_distance = 0.0
    for route in routes:
        inner = [n for n in route if n != 0]
        if not inner:
            continue
        path = [0] + inner + [0]
        for i in range(len(path) - 1):
            raw_distance += dist_matrix[path[i]][path[i + 1]]
    return C * raw_distance
