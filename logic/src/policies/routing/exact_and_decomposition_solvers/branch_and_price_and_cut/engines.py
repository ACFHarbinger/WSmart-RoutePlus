"""Pricing and cut engines of the BPC, chosen by ``BPCConfig.pricing_engine``.

``labelling`` (default) prices with the ng-route labelling RCSPP and separates
every requested cut family plus the limited-memory rank-1 cuts. ``cptp`` prices
with the CPTP branch-and-cut of the ``logic/submodules/cptp`` submodule
(:mod:`...pricing.cptp_pricer`). It keeps only the cut families whose duals that
pricer can express: rounded capacity (with its SECs), knapsack cover, edge
clique, fleet cover and node-based physical-capacity LCIs. Subset-row, rank-1,
multistar and arc-saturation LCI cuts are not separated.

Attributes:
    PRICING_ENGINES: The accepted ``pricing_engine`` values.
    CPTP_CUT_FAMILIES: The cut families separated with the CPTP pricer.
    make_pricing_solver: Build the configured pricer.
    make_cut_engine: Build the configured cut engine.

Example:
    >>> pricer = make_pricing_solver(cfg, k_nodes, dist, wastes, capacity, R, C)
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from logic.src.configs.policies.bpc import BPCConfig
from logic.src.policies.helpers.solvers_and_matheuristics.pricing import RCSPPSolver
from logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes import (
    CompositeCuttingPlaneEngine,
    CuttingPlaneEngine,
    LimitedMemoryRank1CutEngine,
    create_cutting_plane_engine,
)

PRICING_ENGINES = ("labelling", "cptp")
CPTP_CUT_FAMILIES = ("rcc", "cover", "edge_clique", "fleet_cover", "physical_lci")


def _engine_name(cfg: BPCConfig) -> str:
    name = str(cfg.pricing_engine).lower()
    if name not in PRICING_ENGINES:
        raise ValueError(f"Unknown BPC pricing_engine {cfg.pricing_engine!r}; valid: {list(PRICING_ENGINES)}")
    return name


def make_pricing_solver(
    cfg: BPCConfig,
    k_nodes: int,
    dist: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
) -> RCSPPSolver:
    """The pricing solver of ``cfg.pricing_engine``."""
    kwargs: Dict[str, Any] = dict(
        use_ng_routes=cfg.use_ng_routes,
        ng_neighborhood_size=min(cfg.ng_neighborhood_size, k_nodes - 1),
        timeout=cfg.rcspp_timeout,
        max_labels=cfg.rcspp_max_labels,
    )
    if _engine_name(cfg) == "cptp":
        from logic.src.policies.helpers.solvers_and_matheuristics.pricing.cptp_pricer import CPTPPricingSolver

        return CPTPPricingSolver(
            k_nodes,
            dist,
            wastes,
            capacity,
            R,
            C,
            threads=cfg.cptp_threads,
            options=dict(cfg.cptp_options or {}),
            **kwargs,
        )
    return RCSPPSolver(k_nodes, dist, wastes, capacity, R, C, **kwargs)


def make_cut_engine(cfg: BPCConfig, v_model: Any, sep_engine: Any, dist: np.ndarray) -> CuttingPlaneEngine:
    """The cut engine of ``cfg.cutting_planes`` for ``cfg.pricing_engine``."""
    name = cfg.cutting_planes
    if _engine_name(cfg) == "cptp":
        families = CPTP_CUT_FAMILIES if name in ("all", "composite") else (name,)
        unknown = [f for f in families if f not in CPTP_CUT_FAMILIES]
        if unknown:
            raise ValueError(
                f"cutting_planes {name!r} has no dual form in the CPTP pricer; use one of {CPTP_CUT_FAMILIES}"
            )
        return CompositeCuttingPlaneEngine(
            [
                create_cutting_plane_engine(f, v_model, sep_engine, dist_matrix=dist, route_budget=cfg.route_budget)
                for f in families
            ]
        )
    cut_engine = create_cutting_plane_engine(name, v_model, sep_engine, dist_matrix=dist, route_budget=cfg.route_budget)
    # The rank-1 cuts address the profit-selection x routing duality and are always on;
    # "all" already includes them.
    if name not in ("all", "composite"):
        cut_engine = CompositeCuttingPlaneEngine([cut_engine, LimitedMemoryRank1CutEngine(v_model, max_subset_size=5)])
    return cut_engine


__all__ = ["PRICING_ENGINES", "CPTP_CUT_FAMILIES", "make_pricing_solver", "make_cut_engine"]
