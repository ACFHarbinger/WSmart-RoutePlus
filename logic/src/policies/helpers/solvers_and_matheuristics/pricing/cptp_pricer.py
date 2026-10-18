"""CPTP branch-and-cut pricing for the BPC master (``pricing_engine: cptp``).

The BPC pricing problem is a capacitated profitable tour problem (CPTP): find
one tour from the depot, within the vehicle capacity, that maximises collected
reduced profit minus travel cost. This engine solves it with the open
branch-and-cut of Spoorendonk (2026, arXiv:2607.04497), which re-implements
Jepsen et al. (2014, Discrete Optimization 14:78-96) on HiGHS; the solver is
the ``logic/submodules/cptp`` submodule, imported as the ``cptp`` package.

The master's duals map onto the CPTP data as follows:

- node coverage duals and node-based lifted cover duals lower the node profits;
- rounded capacity cut duals add to every edge crossing the cut's set, and
  edge-clique duals to their edge;
- the fleet dual is a constant per route.

Subset-row, rank-1, multistar and arc-saturation LCI duals have no CPTP form.
Neither do must-use arcs, Ryan-Foster branches or Farkas pricing. Whenever one
is active, the call falls back to the labelling algorithm. Forbidden arcs drop
an edge only when both directions are forbidden. The returned tour is then
oriented so it avoids the remaining forbidden arcs; if neither orientation does,
the call falls back as well. CPTP returns one tour per call, so a tour that is
already a master column (one at its upper bound keeps a positive reduced cost)
also falls back, letting the labelling return the next-best columns.

CPTP is undirected. An asymmetric distance matrix is priced on its symmetric
part ``(d + d^T) / 2``; the route is then oriented in the cheaper true direction
and its reduced cost recomputed exactly. On a symmetric matrix an optimal CPTP
solve is an exact pricing step; on an asymmetric one it is a heuristic step.

Attributes:
    CPTPPricingSolver: RCSPP solver whose exact step is the CPTP branch-and-cut.

Example:
    >>> pricer = CPTPPricingSolver(n_nodes, dist, wastes, capacity, R, C)
    >>> routes = pricer.solve(dual_values=master.get_reduced_cost_coefficients())
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.helpers.solvers_and_matheuristics.common.route import Route
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.pricing_types import (
    PricingStatus,
    _LCICoverItem,
    _MultistarItem,
)
from logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver import RCSPPSolver

logger = logging.getLogger(__name__)

_DUAL_EPS = 1e-8


def _import_cptp() -> Any:
    try:
        import cptp
    except ImportError as exc:  # pragma: no cover - depends on the optional build
        raise ImportError(
            "pricing_engine='cptp' needs the cptp package: build it from the logic/submodules/cptp "
            "submodule (uv pip install logic/submodules/cptp)."
        ) from exc
    return cptp


class CPTPPricingSolver(RCSPPSolver):
    """RCSPP pricing whose exact step is the CPTP branch-and-cut.

    Attributes:
        threads: Threads the CPTP solver may use per call.
        symmetric: Whether the distance matrix is symmetric (exact pricing).
        stats: Counter of CPTP calls, their outcomes and labelling fallbacks by reason.
        is_known_column: Predicate on a node tuple: already a master column. The engine
            sets it once the master exists.
    """

    def __init__(self, *args: Any, threads: int = 1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cptp = _import_cptp()
        self.threads = max(1, int(threads))
        d = np.asarray(self.cost_matrix, dtype=float)[: self.n_nodes + 1, : self.n_nodes + 1]
        self.symmetric = bool(np.allclose(d, d.T, rtol=1e-9, atol=1e-9))
        iu = np.triu_indices(self.n_nodes + 1, 1)
        self._edges = np.stack(iu, axis=1).astype(np.int32)
        self._base_cost = ((d + d.T) / 2.0)[iu] * self.C
        self._edge_index = {(int(i), int(j)): k for k, (i, j) in enumerate(self._edges)}
        self._demands = np.array([0.0] + [float(self.wastes.get(i, 0.0)) for i in range(1, self.n_nodes + 1)])
        self.stats: Counter = Counter()
        self.is_known_column: Callable[[Tuple[int, ...]], bool] = lambda _sig: False

    # The labelling DP is replaced by a CPTP solve whenever the call is representable.
    def _label_correcting_algorithm(  # type: ignore[override]
        self,
        max_routes: int,
        forbidden_arcs: FrozenSet[Tuple[int, int]],
        required_successors: Dict[int, int],
        required_predecessors: Dict[int, int],
        rf_separate: Set[Tuple[int, int]],
        rf_together: Set[Tuple[int, int]],
        rcc_duals: Dict[FrozenSet[int], float],
        active_sri_subsets: List[FrozenSet[int]],
        sri_dual_values: List[float],
        node_to_sri: Dict[int, List[int]],
        edge_clique_duals: Optional[Dict[Tuple[int, int], float]] = None,
        lci_cover_items: Optional[List[_LCICoverItem]] = None,
        multistar_items: Optional[List[_MultistarItem]] = None,
        exact_mode: bool = False,
    ) -> List[Route]:
        args = (max_routes, forbidden_arcs, required_successors, required_predecessors, rf_separate, rf_together)
        rest = (rcc_duals, active_sri_subsets, sri_dual_values, node_to_sri)
        kw = dict(
            edge_clique_duals=edge_clique_duals,
            lci_cover_items=lci_cover_items,
            multistar_items=multistar_items,
            exact_mode=exact_mode,
        )
        lci_cover_items = lci_cover_items or []
        reason = self._unrepresentable(
            required_successors, rf_separate, rf_together, sri_dual_values, lci_cover_items, multistar_items or []
        )
        if reason is None:
            routes = self._solve_cptp(forbidden_arcs, rcc_duals, edge_clique_duals or {}, lci_cover_items, exact_mode)
            if routes is not None:
                return routes
            reason = "known_column" if self._last_known else "no_valid_orientation"
        self.stats[f"fallback_{reason}"] += 1
        return super()._label_correcting_algorithm(*args, *rest, **kw)  # type: ignore[arg-type]

    def _unrepresentable(
        self,
        required_successors: Dict[int, int],
        rf_separate: Set[Tuple[int, int]],
        rf_together: Set[Tuple[int, int]],
        sri_dual_values: List[float],
        lci_cover_items: List[_LCICoverItem],
        multistar_items: List[_MultistarItem],
    ) -> Optional[str]:
        """Name of the first active element CPTP cannot express, or None."""
        if self.is_farkas:
            return "farkas"
        if required_successors:
            return "must_use_arc"
        if rf_separate or rf_together:
            return "ryan_foster"
        if any(v > _DUAL_EPS for v in sri_dual_values):
            return "sri_dual"
        if any(arc is not None and dual > _DUAL_EPS for _s, _a, dual, arc in lci_cover_items):
            return "arc_lci_dual"
        if any(dual > _DUAL_EPS for _s, dual in multistar_items):
            return "multistar_dual"
        return None

    def _node_lci_penalty(self, node: int, lci_cover_items: List[_LCICoverItem]) -> float:
        total = 0.0
        for cover_set, alphas, dual, arc in lci_cover_items:
            if arc is None and dual > _DUAL_EPS:
                total += alphas.get(node, 1.0 if node in cover_set else 0.0) * dual
        return total

    def _solve_cptp(
        self,
        forbidden_arcs: FrozenSet[Tuple[int, int]],
        rcc_duals: Dict[FrozenSet[int], float],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_cover_items: List[_LCICoverItem],
        exact_mode: bool,
    ) -> Optional[List[Route]]:
        """One CPTP solve; None when its tour has no valid orientation or is already a column."""
        self._last_known = False
        n = self.n_nodes
        profits = np.zeros(n + 1)
        for i in range(1, n + 1):
            profits[i] = self.node_reduced_costs.get(i, 0.0) - self._node_lci_penalty(i, lci_cover_items)

        costs = self._base_cost.copy()
        for node_set, dual in rcc_duals.items():
            if dual <= _DUAL_EPS:
                continue
            inside = np.zeros(n + 1, dtype=bool)
            inside[[v for v in node_set if 0 <= v <= n]] = True
            costs += dual * (inside[self._edges[:, 0]] != inside[self._edges[:, 1]])
        for (u, v), dual in edge_clique_duals.items():
            k = self._edge_index.get((min(u, v), max(u, v)))
            if k is not None:
                costs[k] += dual

        keep = np.ones(len(costs), dtype=bool)
        for u, v in forbidden_arcs:
            if (v, u) in forbidden_arcs:
                k = self._edge_index.get((min(u, v), max(u, v)))
                if k is not None:
                    keep[k] = False

        budget = max(0.05, float(getattr(self, "_current_timeout", self.timeout) or self.timeout))
        t0 = time.perf_counter()
        result = self._cptp.solve(
            num_nodes=n + 1,
            edges=self._edges[keep],
            edge_costs=costs[keep],
            profits=profits,
            demands=self._demands,
            capacity=float(self.capacity),
            time_limit=budget,
            num_threads=self.threads,
        )
        self.stats["cptp_calls"] += 1
        self.stats["cptp_seconds"] += time.perf_counter() - t0
        optimal = bool(result.is_optimal())
        self._timed_out = not optimal
        self.stats["cptp_optimal" if optimal else "cptp_not_optimal"] += 1
        if not result.has_solution():
            self.pricing_status = PricingStatus.TIMED_OUT
            self.last_max_rc = float("inf")
            return []

        tour = [int(v) for v in result.tour]
        nodes = [v for v in tour[1:-1] if v != 0]
        route = self._oriented_route(nodes, forbidden_arcs, rcc_duals, edge_clique_duals, lci_cover_items)
        self._last_known = route is not None and self.is_known_column(tuple(route.nodes))
        if route is None or self._last_known:
            return None

        exact = optimal and self.symmetric
        if exact:
            self.pricing_status = PricingStatus.EXHAUSTIVE if exact_mode else PricingStatus.HEURISTIC
            self.last_max_rc = float(route.reduced_cost)  # type: ignore[arg-type]
        else:
            self.pricing_status = PricingStatus.TIMED_OUT if not optimal else PricingStatus.HEURISTIC
            bound_rc = -float(result.bound) - self.vehicle_dual if self.symmetric else float(route.reduced_cost)
            self.last_max_rc = max(float(route.reduced_cost), bound_rc)  # type: ignore[arg-type]
        return [route] if route.reduced_cost > 0 else []  # type: ignore[operator]

    def _oriented_route(
        self,
        nodes: List[int],
        forbidden_arcs: FrozenSet[Tuple[int, int]],
        rcc_duals: Dict[FrozenSet[int], float],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_cover_items: List[_LCICoverItem],
    ) -> Optional[Route]:
        """The cheaper orientation of the tour that avoids every forbidden arc, with its exact reduced cost."""
        best: Optional[Route] = None
        for seq in (nodes, nodes[::-1]):
            path = [0] + seq + [0]
            if any((a, b) in forbidden_arcs for a, b in zip(path, path[1:], strict=False)):
                continue
            route = self._compute_route_details(seq)
            route.reduced_cost = self._reduced_cost(route, rcc_duals, edge_clique_duals, lci_cover_items)
            if best is None or route.reduced_cost > best.reduced_cost:  # type: ignore[operator]
                best = route
        return best

    def _reduced_cost(
        self,
        route: Route,
        rcc_duals: Dict[FrozenSet[int], float],
        edge_clique_duals: Dict[Tuple[int, int], float],
        lci_cover_items: List[_LCICoverItem],
    ) -> float:
        """Exact reduced cost on the true (directed) costs, as the master computes it."""
        rc = sum(self.node_reduced_costs.get(v, 0.0) for v in route.nodes) - route.cost - self.vehicle_dual
        path = [0] + list(route.nodes) + [0]
        legs = list(zip(path, path[1:], strict=False))
        for node_set, dual in rcc_duals.items():
            if dual > _DUAL_EPS:
                rc -= dual * sum(1 for a, b in legs if (a in node_set) != (b in node_set))
        for a, b in legs:
            rc -= edge_clique_duals.get((min(a, b), max(a, b)), 0.0)
        rc -= sum(self._node_lci_penalty(v, lci_cover_items) for v in route.nodes)
        return rc


__all__ = ["CPTPPricingSolver"]
