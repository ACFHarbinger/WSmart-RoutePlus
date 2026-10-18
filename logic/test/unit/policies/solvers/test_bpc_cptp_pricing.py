"""BPC with the CPTP branch-and-cut pricer (``pricing_engine: cptp``)."""

import itertools

import numpy as np
import pytest
from logic.src.policies.helpers.solvers_and_matheuristics.pricing import RCSPPSolver
from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.bpc_engine import run_bpc
from logic.src.policies.routing.exact_and_decomposition_solvers.branch_and_price_and_cut.engines import (
    make_cut_engine,
)

pytest.importorskip("cptp")
pytest.importorskip("gurobipy")

from logic.src.policies.helpers.solvers_and_matheuristics.pricing.cptp_pricer import (  # noqa: E402
    CPTPPricingSolver,
)

pytestmark = [pytest.mark.unit]


def _instance(seed, n, asym=False):
    rng = np.random.default_rng(seed)
    xy = rng.random((n + 1, 2)) * 10
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    if asym:
        d = d * rng.uniform(0.8, 1.2, d.shape)
        np.fill_diagonal(d, 0.0)
    wastes = {i: float(rng.uniform(5, 40)) for i in range(1, n + 1)}
    duals = {i: float(rng.uniform(-5, 40)) for i in range(1, n + 1)}
    return d, wastes, duals, float(rng.uniform(0, 5))


def _best_route_rc(n, d, wastes, cap, duals, v_dual):
    best = -float("inf")
    for k in range(1, n + 1):
        for subset in itertools.combinations(range(1, n + 1), k):
            if sum(wastes[i] for i in subset) > cap:
                continue
            for perm in itertools.permutations(subset):
                path = (0,) + perm + (0,)
                cost = sum(d[a, b] for a, b in zip(path, path[1:], strict=False))
                best = max(best, sum(wastes[i] - duals[i] for i in subset) - cost - v_dual)
    return best


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("asym", [False, True])
def test_cptp_pricer_matches_brute_force_and_labelling(seed, asym):
    n, cap = 6, 70.0
    d, wastes, duals, v_dual = _instance(seed, n, asym)
    dual_values = {"node_duals": {**duals, "vehicle_limit": v_dual}, "rcc_duals": {}, "sri_duals": {}}
    optimum = _best_route_rc(n, d, wastes, cap, duals, v_dual)
    cptp_pricer = CPTPPricingSolver(n, d, wastes, cap, 1.0, 1.0, use_ng_routes=False, timeout=10)
    routes = cptp_pricer.solve(dual_values=dual_values, max_routes=5, exact_mode=True)
    labelling = RCSPPSolver(n, d, wastes, cap, 1.0, 1.0, use_ng_routes=False, timeout=10)
    expected = labelling.solve(dual_values=dual_values, max_routes=5, exact_mode=True)
    assert cptp_pricer.stats["cptp_calls"] == 1
    if optimum <= 0:
        assert routes == []
        return
    if asym:
        # Priced on the symmetric part: a valid column, never better than the optimum.
        assert routes and 0 < routes[0].reduced_cost <= optimum + 1e-6
    else:
        assert routes[0].reduced_cost == pytest.approx(optimum, abs=1e-6)
        assert expected[0].reduced_cost == pytest.approx(optimum, abs=1e-6)


def test_cptp_pricer_falls_back_on_unrepresentable_duals():
    n, cap = 6, 70.0
    d, wastes, duals, _ = _instance(0, n)
    pricer = CPTPPricingSolver(n, d, wastes, cap, 1.0, 1.0, use_ng_routes=False, timeout=10)
    pricer.solve(
        dual_values={"node_duals": duals, "rcc_duals": {}, "sri_duals": {frozenset({1, 2, 3}): 2.0}},
        exact_mode=True,
    )
    assert pricer.stats["fallback_sri_dual"] == 1
    assert pricer.stats["cptp_calls"] == 0


def test_cptp_engine_keeps_only_cut_families_it_prices():
    from logic.src.configs.policies.bpc import BPCConfig

    engine = make_cut_engine(BPCConfig(pricing_engine="cptp"), v_model=None, sep_engine=object(), dist=np.zeros((2, 2)))
    names = {type(e).__name__ for e in engine.engines}
    assert "SubsetRowCutEngine" not in names and "LimitedMemoryRank1CutEngine" not in names
    assert "RoundedCapacityCutEngine" in names
    with pytest.raises(ValueError):
        make_cut_engine(BPCConfig(pricing_engine="cptp", cutting_planes="sri"), None, object(), np.zeros((2, 2)))


def test_bpc_with_cptp_pricing_reaches_the_canonical_optimum():
    dist = np.array([[0, 10, 10, 20], [10, 0, 5, 15], [10, 5, 0, 10], [20, 15, 10, 0]], dtype=float)
    wastes = {1: 5.0, 2: 5.0, 3: 5.0}
    cfg = {"max_cg_iterations": 50, "max_bb_nodes": 200, "optimality_gap": 1e-9, "pricing_engine": "cptp"}
    routes, profit, _cost = run_bpc(build_subproblem(dist, wastes, 10.0, 10.0, 1.0, mandatory_nodes=[3]), cfg)
    labelled = run_bpc(
        build_subproblem(dist, wastes, 10.0, 10.0, 1.0, mandatory_nodes=[3]), {**cfg, "pricing_engine": "labelling"}
    )
    assert profit == pytest.approx(labelled.profit, abs=1e-6)
    assert any(3 in r for r in routes)
