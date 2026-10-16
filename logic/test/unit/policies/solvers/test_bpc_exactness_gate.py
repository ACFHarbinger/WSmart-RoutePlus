"""Exact-solver regression gate for P-33 (RQ-kimi-1).

On tiny independently checkable instances, both exact engines must return an
incumbent equal to the brute-force optimum within 1e-6 and log a certified
Lagrangian upper bound (z_UB) at or above the incumbent — i.e. valid dual
certificates, per the owner ruling on RQ-kimi-1 (equality of incumbent
objective and certified bound within 1e-6, not bit-identical routes).
"""

import re

import numpy as np
import pytest
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import (
    run_bpc,
)
from logic.src.policies.routing.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine import (
    run_ms_bpc_sp,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]

_ZUB_RE = re.compile(r"z_UB=([-\d.eE+]+)")


def _has_gurobi_license() -> bool:
    try:
        import gurobipy as _gp

        _m = _gp.Model("_check")
        _m.dispose()
        return True
    except Exception:
        return False


_needs_license = pytest.mark.skipif(
    not _has_gurobi_license(),
    reason="Requires a valid Gurobi license",
)


def _partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _partitions(rest):
        yield [[first]] + p
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1 :]


def _route_cost(dist_matrix, nodes):
    import itertools

    best = float("inf")
    for perm in itertools.permutations(nodes):
        path = (0,) + perm + (0,)
        best = min(best, sum(dist_matrix[a][b] for a, b in zip(path, path[1:], strict=False)))
    return best


def _brute_force_optimum(dist_matrix, wastes, capacity, R, C, vehicle_limit=None):
    n = len(wastes)
    ids = list(range(1, n + 1))
    best = -float("inf")
    for r in range(0, n + 1):
        for subset in __import__("itertools").combinations(ids, r):
            load = sum(wastes[i] for i in subset)
            for part in _partitions(list(subset)):
                if vehicle_limit is not None and len(part) > vehicle_limit:
                    continue
                if any(sum(wastes[i] for i in route) > capacity + 1e-9 for route in part):
                    continue
                km = sum(_route_cost(dist_matrix, route) for route in part)
                best = max(best, R * load - C * km)
    return best


def _certified_bound(caplog, incumbent, optimum):
    """Require a valid certificate: a logged z_UB at/above the incumbent, or —
    when the instance converges trivially without CG iterations (no bound
    line) — exact agreement with the externally computed brute-force optimum.
    """
    ubs = [float(m.group(1)) for m in _ZUB_RE.finditer(caplog.text)]
    if ubs:
        best_ub = min(ubs)
        assert best_ub >= incumbent - 1e-6, (
            f"certified bound {best_ub} is below the incumbent {incumbent} — invalid certificate"
        )
    else:
        assert abs(incumbent - optimum) <= 1e-9, (
            "no z_UB was logged and the incumbent does not exactly match the "
            f"brute-force optimum ({incumbent} vs {optimum})"
        )


# Canonical 3-node fixture (WP-04/WP-11): needs >= 2 vehicles at capacity 10.
CANONICAL = dict(
    dist=np.array([[0, 10, 10, 20], [10, 0, 5, 15], [10, 5, 0, 10], [20, 15, 10, 0]]),
    wastes={1: 5.0, 2: 5.0, 3: 5.0},
    capacity=10.0,
    R=10.0,
    C=1.0,
)


@_needs_license
def test_bpc_matches_brute_force_with_valid_certificate(caplog):
    caplog.set_level("INFO")
    optimum = _brute_force_optimum(
        CANONICAL["dist"].tolist(), CANONICAL["wastes"], CANONICAL["capacity"], CANONICAL["R"], CANONICAL["C"]
    )
    routes, profit, _cost = run_bpc(
        build_subproblem(CANONICAL["dist"], CANONICAL["wastes"], CANONICAL["capacity"], CANONICAL["R"], CANONICAL["C"]),
        {"max_cg_iterations": 50, "max_bb_nodes": 200, "optimality_gap": 1e-9},
    )
    assert routes, "BPC returned no routes on the canonical instance"
    assert abs(profit - optimum) <= 1e-6 * max(1.0, abs(optimum)), (
        f"BPC incumbent {profit} != brute-force optimum {optimum}"
    )
    _certified_bound(caplog, profit, optimum)
    from logic.src.pipeline.simulations.solver_status import current_solver_certificate

    proof = current_solver_certificate()
    assert proof is not None
    assert proof["incumbent"] == profit
    assert proof["bound"] >= optimum - 1e-6
    assert proof["source"] == "root_cg_exhausted"


@_needs_license
def test_ms_bpc_sp_matches_brute_force_with_valid_certificate(caplog):
    caplog.set_level("INFO")
    optimum = _brute_force_optimum(
        CANONICAL["dist"].tolist(), CANONICAL["wastes"], CANONICAL["capacity"], CANONICAL["R"], CANONICAL["C"]
    )
    routes, profit, _cost = run_ms_bpc_sp(
        build_subproblem(CANONICAL["dist"], CANONICAL["wastes"], CANONICAL["capacity"], CANONICAL["R"], CANONICAL["C"]),
        {"max_cg_iterations": 50, "max_bb_nodes": 200, "optimality_gap": 1e-9},
    )
    assert routes, "MS-BPC-SP returned no routes on the canonical instance"
    assert abs(profit - optimum) <= 1e-6 * max(1.0, abs(optimum)), (
        f"MS-BPC-SP incumbent {profit} != brute-force optimum {optimum}"
    )
    _certified_bound(caplog, profit, optimum)


@_needs_license
def test_ms_bpc_sp_fleet_limited_matches_brute_force_with_valid_certificate(caplog):
    caplog.set_level("INFO")
    dist = np.array(
        [
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 2.0, 2.0, 2.0],
            [1.0, 2.0, 0.0, 2.0, 2.0],
            [1.0, 2.0, 2.0, 0.0, 2.0],
            [1.0, 2.0, 2.0, 2.0, 0.0],
        ]
    )
    wastes = {1: 6.0, 2: 6.0, 3: 6.0, 4: 6.0}
    optimum = _brute_force_optimum(dist.tolist(), wastes, 10.0, 10.0, 1.0, vehicle_limit=2)
    routes, profit, _cost = run_ms_bpc_sp(
        build_subproblem(dist, wastes, 10.0, 10.0, 1.0, n_vehicles=2),
        {
            "optimality_gap": 1e-9,
            "early_termination_gap": 1e-9,
            "exact_mode": True,
            "max_bb_nodes": 1000,
            "time_limit": 60,
            "enable_strong_branching_heuristic": False,
        },
    )
    assert len(routes) <= 2, f"fleet limit violated: {len(routes)} routes"
    assert abs(profit - optimum) <= 1e-6 * max(1.0, abs(optimum)), (
        f"MS-BPC-SP incumbent {profit} != brute-force optimum {optimum}"
    )
    _certified_bound(caplog, profit, optimum)
