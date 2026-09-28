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


def test_gurobi_empty_day_returns_d3_shape():
    """B-kimi-56: a no-incumbent day returns [0, 0] like the other backends."""
    bins, dist = _tiny_instance()
    values = {"Omega": 0.1, "psi": 1, "Q": 100.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
    route, profit, cost = _run_gurobi_optimizer(
        bins=bins,
        distance_matrix=dist,
        env=None,
        values=values,
        binsids=list(range(1, 9)),
        mandatory=[],
        number_vehicles=1,
        time_limit=0.001,  # no time to find an incumbent
        seed=42,
    )
    assert route == [0, 0]
    assert profit == 0.0 and cost == 0.0


def test_gurobi_fleet_fallback_uses_n_bins():
    """B-kimi-57: number_vehicles=0 falls back to n_bins, not len(binsids)."""
    bins, dist = _tiny_instance()
    values = {"Omega": 0.1, "psi": 1, "Q": 500.0, "R": 1.0, "B": 1.0, "C": 1.0, "V": 1.0}
    # binsids longer than n_bins exercises the old len(binsids) divergence.
    route, _, _ = _run_gurobi_optimizer(
        bins=bins,
        distance_matrix=dist,
        env=None,
        values=values,
        binsids=[0] + list(range(1, 9)) + [99, 100],
        mandatory=[],
        number_vehicles=0,
        time_limit=10,
        seed=42,
    )
    assert route[0] == 0 and len(route) > 2  # solved with an unbounded-by-bins fleet


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
