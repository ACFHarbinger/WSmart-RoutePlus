"""Regressions for the SWC-TCF backend-difference rows (B-kimi-55..59)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pytest
from logic.src.policies.route_construction.base.base_routing_policy import BaseRoutingPolicy
from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow import (
    policy_swc_tcf,
)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi import (
    _run_gurobi_optimizer,
)

pytestmark = [pytest.mark.unit]

gp = pytest.importorskip("gurobipy")


def _tiny_instance() -> tuple:
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, size=(9, 2))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    bins = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    return bins, dist.tolist()


def test_gurobi_no_incumbent_day_is_deterministic(monkeypatch):
    """B-kimi-56 / Codex review #6: a no-incumbent day returns [0, 0].

    Deterministic: the zero-incumbent condition is mocked at the solver
    boundary (no reliance on a 1 ms budget racing the trivial incumbent).
    """
    from unittest.mock import MagicMock

    from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow import (
        gurobi as gurobi_mod,
    )

    class _AnyExpr:
        """Arbitrary arithmetic/comparison expression: every operation yields itself."""

        def __add__(self, other):
            return self

        def __radd__(self, other):
            return self

        def __sub__(self, other):
            return self

        def __rsub__(self, other):
            return self

        def __mul__(self, other):
            return self

        def __rmul__(self, other):
            return self

        def __le__(self, other):
            return self

        def __ge__(self, other):
            return self

        def __eq__(self, other):
            return self

        def __hash__(self):
            return id(self)

    class _VarDict(dict):
        def __missing__(self, key):
            self[key] = _AnyExpr()
            return self[key]

    fake_model = MagicMock()
    fake_model.Params.Threads = 0
    fake_model.Params.SoftMemLimit = float("inf")
    fake_model.SolCount = 0
    fake_model.Status = gurobi_mod.GRB.TIME_LIMIT
    fake_model.addVars.side_effect = lambda *a, **k: _VarDict()
    fake_model.addVar.side_effect = lambda *a, **k: _AnyExpr()
    monkeypatch.setattr(gurobi_mod.gp, "Model", lambda *a, **k: fake_model)

    bins, dist = _tiny_instance()
    values = {"Omega": 0.1, "psi": 1, "Q": 100.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
    route, profit, cost = gurobi_mod._run_gurobi_optimizer(
        bins=bins,
        distance_matrix=dist,
        env=None,
        values=values,
        binsids=list(range(1, 9)),
        mandatory=[],
        number_vehicles=1,
        time_limit=60,
        seed=42,
    )
    assert fake_model.optimize.called
    assert route == [0, 0]
    assert profit == 0.0 and cost == 0.0


def test_gurobi_empty_optimal_plan_returns_d3_shape():
    """B-kimi-56: when the optimum collects nothing, the day is [0, 0].

    Deterministic real solve: zero revenue makes any collection a pure cost,
    so the optimal plan is empty and the empty-walk normalisation applies.
    """
    bins, dist = _tiny_instance()
    values = {"Omega": 0.1, "psi": 1, "Q": 100.0, "R": 0.0, "B": 1.0, "C": 1.0, "V": 1.0}
    route, profit, cost = _run_gurobi_optimizer(
        bins=bins,
        distance_matrix=dist,
        env=None,
        values=values,
        binsids=list(range(1, 9)),
        mandatory=[],
        number_vehicles=1,
        time_limit=30,
        seed=42,
    )
    assert route == [0, 0]
    assert profit == 0.0 and cost == 0.0


def test_gurobi_unbounded_fleet_serves_capacity_requiring_witness():
    """B-kimi-57/58 / Codex review #6: a witness that NEEDS two vehicles.

    Four bins at 100 % with Q = 250 % cannot fit in one route; collecting all
    is optimal, so an unbounded fleet must return a two-route day. A silent
    one-vehicle cap (the pre-B-kimi-58 default) drops a bin and returns a
    single route, so this fails on the prior implementation.
    """
    bins = np.array([100.0, 100.0, 100.0, 100.0])
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 10, size=(5, 2))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)).tolist()
    values = {"Omega": 0.1, "psi": 1, "Q": 250.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
    route, profit, _ = _run_gurobi_optimizer(
        bins=bins,
        distance_matrix=dist,
        env=None,
        values=values,
        binsids=[0, 11, 12, 13, 14],
        mandatory=[],
        number_vehicles=0,
        time_limit=30,
        seed=42,
    )
    assert route != [0, 0]
    assert route[0] == 0 and route[-1] == 0
    assert {n for n in route if n != 0} == {11, 12, 13, 14}, "a bin was dropped despite the unbounded fleet"
    interior = [i for i, n in enumerate(route) if n == 0]
    assert len(interior) >= 3, f"expected >= 2 depot-delimited routes, got {route}"
    # each route carries at most Q percent
    segs, cur = [], []
    for n in route[1:]:
        if n == 0:
            segs.append(cur)
            cur = []
        else:
            cur.append(n)
    assert all(len(seg) * 100.0 <= 250.0 + 1e-6 for seg in segs if seg)
    assert len([seg for seg in segs if seg]) >= 2, f"expected two non-empty routes, got {segs}"

    # The bound is enforced, not ignored: capped at one vehicle the same
    # instance must drop a bin and return a single route.
    capped, _, _ = _run_gurobi_optimizer(
        bins=bins,
        distance_matrix=dist,
        env=None,
        values=values,
        binsids=[0, 11, 12, 13, 14],
        mandatory=[],
        number_vehicles=1,
        time_limit=30,
        seed=42,
    )
    capped_segs, cur = [], []
    for n in capped[1:]:
        if n == 0:
            if cur:
                capped_segs.append(cur)
            cur = []
        else:
            cur.append(n)
    assert len([seg for seg in capped_segs if seg]) == 1, f"one-vehicle cap not enforced: {capped}"
    assert {n for n in capped if n != 0} != {11, 12, 13, 14}


def test_policy_splits_depot_delimited_flat_route(monkeypatch):
    """B-kimi-58: the adapter splits a depot-delimited flat list into routes."""
    captured: Dict[str, Any] = {}

    def fake_optimizer(*, bins, distance_matrix, values, binsids, mandatory_nodes, number_vehicles, **kw):
        captured["number_vehicles"] = number_vehicles
        return [0, 11, 12, 0, 13, 0], 5.0, 2.0

    monkeypatch.setattr(policy_swc_tcf, "run_swc_tcf_optimizer", fake_optimizer)

    class _Params:
        time_limit = 60.0
        framework = "gurobi"
        engine = "gurobi"

    policy = policy_swc_tcf.SWCTCFPolicy.__new__(policy_swc_tcf.SWCTCFPolicy)
    sub_dist = np.zeros((4, 4))
    routes, profit, cost = policy._run_solver(
        sub_dist_matrix=sub_dist,
        sub_wastes={1: 1.0, 2: 1.0, 3: 1.0},
        capacity=100.0,
        revenue=1.0,
        cost_unit=1.0,
        values={},
        mandatory_nodes=[],
        params=_Params(),
        n_vehicles=0,
    )
    assert routes == [[11, 12], [13]]
    assert profit == 5.0 and cost == 2.0
    assert captured["number_vehicles"] == 0  # unbounded, not the old silent 1


@dataclass
class _FakeConfig:
    capacity: float = 100.0
    revenue: float = 0.5
    cost_unit: float = 1.0
    density: float = 1.0
    bin_volume: float = 1.0
    seed: Optional[int] = None


class _TypedPolicy(BaseRoutingPolicy):
    """Minimal adapter to exercise the typed-config path of load_config."""

    _config = _FakeConfig()

    def execute(self, **kwargs):
        raise NotImplementedError

    def _run_solver(self, *args, **kwargs):
        raise NotImplementedError

    def _get_config_key(self) -> str:
        return "typed_fake"


def test_typed_path_flattens_engine_nested_overrides(monkeypatch):
    """B-kimi-59: engine-nested runtime overrides reach the values dict on the typed path."""
    monkeypatch.setattr(
        "logic.src.policies.route_construction.base.base_routing_policy.load_area_and_waste_type_params",
        lambda area, wt: (100.0, 0.5, 1.0, 1.0, 1.0),
    )
    policy = _TypedPolicy(config=_FakeConfig())
    raw_cfg = {
        "typed_fake": [
            {"gurobi": [{"time_limit": 7.0}, {"vrpp": False}]},
            {"seed": 123},
        ]
    }
    capacity, revenue, cost_unit, values = policy._load_area_params("rio", "plastic", raw_cfg)
    assert values["time_limit"] == 7.0, "engine-nested override was dropped by the typed path"
    assert values["vrpp"] is False
    assert values["seed"] == 123
    assert capacity == 100.0


def test_route_extraction_parity_across_backends(capsys):
    """M-kimi-02: one shared extractor — all three backends must agree on a fixed instance."""
    from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.dispatcher import (
        run_swc_tcf_optimizer,
    )

    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 10, size=(9, 2))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)).tolist()
    bins = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    values = {"Omega": 0.1, "psi": 1, "Q": 250.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}

    results = {}
    for framework, optimizer in [("gurobi", "gurobi"), ("ortools", "SCIP"), ("pyomo", "gurobi")]:
        with capsys.disabled():
            route, profit, _ = run_swc_tcf_optimizer(
                bins=bins,
                distance_matrix=dist,
                values=values,
                binsids=list(range(1, 9)),
                mandatory_nodes=[2, 5],
                number_vehicles=0,
                time_limit=30,
                framework=framework,
                optimizer=optimizer,
                seed=42,
            )
        results[framework] = (route, profit)

    def _bins(flat):
        return frozenset(n for n in flat if n != 0)

    g_route, g_profit = results["gurobi"]
    assert g_route != [0, 0], "gurobi backend found no solution on the parity instance"
    for framework in ("ortools", "pyomo"):
        route, profit = results[framework]
        if route == [0, 0]:
            pytest.skip(f"{framework} backend unavailable in this environment")
        # Visit order may differ among equal-profit optima; profit and the
        # collected bin set must agree.
        assert _bins(route) == _bins(g_route), f"{framework} collected a different bin set"
        assert abs(profit - g_profit) <= 1e-6 * max(1.0, abs(g_profit)), (
            f"{framework} profit {profit} diverges from native gurobi {g_profit}"
        )


def test_build_tcf_data_shared_prep():
    """M-kimi-03: one shared preparation for all backends (values pinned)."""
    from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data import (
        build_tcf_data,
    )

    bins = np.array([10.0, 50.0, 90.0])
    dist = [
        [0.0, 1.0, 7001.0, 7001.0],
        [1.0, 0.0, 2.0, 7001.0],
        [7001.0, 2.0, 0.0, 2.0],
        [7001.0, 7001.0, 2.0, 0.0],
    ]
    values = {"Omega": 0.1, "psi": 1, "Q": 250.0, "R": 0.7, "C": 1.0}
    d = build_tcf_data(bins, dist, values, [0, 11, 12, 13], [12], number_vehicles=0)

    assert d.n_bins == 3 and d.nodes == [0, 1, 2, 3]
    assert d.S_dict == {0: 0.0, 1: 10.0, 2: 50.0, 3: 90.0}
    # bin 12 (local 2) is mandatory; the 7001 km arcs are cut, the 7001-km row/col
    # for node 2 is isolated except the 2 km arc (2, 3) / (3, 2).
    assert d.criticos_dict == {0: False, 1: False, 2: True, 3: False}
    assert (0, 2) not in d.valid_arcs and (2, 0) not in d.valid_arcs
    assert (2, 3) in d.valid_arcs and (1, 2) in d.valid_arcs
    assert d.id_map == {0: 0, 1: 11, 2: 12, 3: 13}
    assert d.max_trucks == 3  # unbounded falls back to n_bins

    capped = build_tcf_data(bins, dist, values, [0, 11, 12, 13], [12], number_vehicles=2)
    assert capped.max_trucks == 2
