"""Regression tests for the CVRP engines: capacity, coverage and engine dispatch."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from logic.src.configs.policies import CVRPConfig
from logic.src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.cvrp import (
    find_routes,
    find_routes_ortools,
)
from logic.src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.params import (
    CVRPParams,
)
from logic.src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp import (
    CVRPPolicy,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

MODULE = "logic.src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp"


@pytest.fixture
def instance():
    rng = np.random.default_rng(0)
    n = 30
    xy = rng.random((n + 1, 2)) * 100
    dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    # Fractional fill levels: integer truncation used to overload PyVRP routes.
    wastes = rng.uniform(50, 99.9, n)
    return np.round(dist * 10).astype(int), wastes, 400.0, np.arange(1, n + 1)


def _routes(tour):
    routes, current = [], []
    for node in tour[1:]:
        if node == 0:
            routes.append(current)
            current = []
        else:
            current.append(node)
    return routes


@pytest.mark.parametrize(
    "solve",
    [
        lambda d, w, c, t: find_routes_ortools(d, w, c, t, 0, time_limit=1, seed=1),
        lambda d, w, c, t: find_routes(d, w, c, t, 0, time_limit=1, seed=1),
        lambda d, w, c, t: find_routes(d, w, c, t, 0, engine="clarke_wright"),
    ],
    ids=["ortools", "pyvrp", "clarke_wright"],
)
def test_engines_respect_capacity_and_visit_every_bin(instance, solve):
    dist, wastes, capacity, to_collect = instance
    tour = solve(dist, wastes, capacity, to_collect)

    assert tour[0] == 0 and tour[-1] == 0
    assert sorted(n for n in tour if n != 0) == list(to_collect)
    for route in _routes(tour):
        assert sum(wastes[i - 1] for i in route) <= capacity


def test_params_default_to_ortools_with_a_seed():
    params = CVRPParams.from_config(CVRPConfig())
    assert params.engine == "ortools"
    assert params.seed == 42
    assert CVRPParams.from_config({"engine": "pyvrp", "seed": None}).seed == 42


@pytest.mark.parametrize(
    ("engine", "solver", "engine_kwarg"),
    [("ortools", "find_routes_ortools", None), ("pyvrp", "find_routes", "pyvrp"), ("clarke_wright", "find_routes", "clarke_wright")],
)
def test_policy_dispatches_engine_and_uses_base_profit_units(engine, solver, engine_kwarg):
    policy = CVRPPolicy({"engine": engine, "time_limit": 1, "seed": 7})
    bins = SimpleNamespace(c=np.array([50.0, 60.0]), n=2)
    dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    with patch(f"{MODULE}.{solver}", return_value=[0, 1, 2, 0]) as mocked, patch.object(
        policy, "_load_area_params", return_value=(400.0, 2.0, 0.5, {})
    ):
        tour, cost, profit, _, _ = policy.execute(mandatory=[1, 2], bins=bins, distance_matrix=dist, seed=7)

    mocked.assert_called_once()
    if engine_kwarg is not None:
        assert mocked.call_args.kwargs["engine"] == engine_kwarg
    # distancesC falls back to the simulator's 0.1 km integer matrix.
    np.testing.assert_array_equal(mocked.call_args.args[0], np.round(dist * 10).astype("int32"))
    assert tour == [0, 1, 2, 0]
    assert cost == pytest.approx(4.5)
    assert profit == pytest.approx((50.0 + 60.0) * 2.0 - 4.5 * 0.5)


def test_policy_rejects_unknown_engine():
    policy = CVRPPolicy({"engine": "gurobi"})
    bins = SimpleNamespace(c=np.array([50.0]), n=1)
    with patch.object(policy, "_load_area_params", return_value=(400.0, 1.0, 1.0, {})), pytest.raises(
        ValueError, match="Unknown CVRP engine"
    ):
        policy.execute(mandatory=[1], bins=bins, distance_matrix=np.zeros((2, 2)), seed=1)
