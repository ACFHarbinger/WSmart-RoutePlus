"""The default SWC-TCF model is the published SWCR model (Ramos et al., 2018, eqs. 6, 8, 11-15, 17-21)."""

import numpy as np
import pytest
from logic.src.configs.policies import SWCTCFConfig
from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi import (
    _options,
    _run_gurobi_optimizer,
)
from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.params import (
    SWCTCFParams,
)

pytestmark = [pytest.mark.unit]

gp = pytest.importorskip("gurobipy")


def _instance(seed: int, n: int = 12):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n + 1, 2))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)).tolist()
    return rng.uniform(5.0, 95.0, n), dist


def _solve(bins, dist, values, mandatory=(3,), number_vehicles=0):
    return _run_gurobi_optimizer(
        bins=bins,
        distance_matrix=dist,
        env=None,
        values=values,
        binsids=list(range(1, len(bins) + 1)),
        mandatory=list(mandatory),
        number_vehicles=number_vehicles,
        time_limit=60,
        seed=1,
    )


def _trips(route):
    trips, current = [], []
    for node in route[1:]:
        if node == 0:
            trips.append(current)
            current = []
        else:
            current.append(node)
    return trips


def test_defaults_are_the_original_algorithm():
    params, config = SWCTCFParams(), SWCTCFConfig()
    for source in (params, config):
        assert source.formulation == "paper"
        assert source.depot_inflow == "equal"
        assert source.solver_tuning is False
        assert source.relax_forced_on_infeasible is False
        assert source.link_depot_arcs is False
        assert source.max_arc_distance_km is None
        assert source.warm_start is False
    assert _options({})["formulation"] == "paper"


@pytest.mark.parametrize(
    "bad", [{"formulation": "mtz"}, {"depot_inflow": "ge"}, {"max_arc_distance_km": 0}, {"max_arc_distance_km": "inf"}]
)
def test_invalid_options_are_rejected(bad):
    with pytest.raises(ValueError):
        _options(bad)


@pytest.mark.parametrize(("seed", "revenue", "fleet"), [(10, 0.04, 0), (11, 0.06, 1), (12, 0.1, 0), (15, 0.1, 1)])
def test_paper_and_directed_models_agree_on_symmetric_instances(seed, revenue, fleet):
    """Same optimum on symmetric distances, including partial selection and a one-truck fleet."""
    bins, dist = _instance(seed)
    values = {"Omega": 0.1, "psi": 1, "Q": 300.0, "R": revenue, "C": 1.0}
    paper = _solve(bins, dist, values, number_vehicles=fleet)
    directed = _solve(bins, dist, {**values, "formulation": "directed"}, number_vehicles=fleet)
    assert paper[1] == pytest.approx(directed[1], abs=1e-4)
    assert sorted(set(paper[0]) - {0}) == sorted(set(directed[0]) - {0})


def test_paper_routes_are_valid_tours():
    bins, dist = _instance(12)
    capacity = 300.0
    route, profit, cost = _solve(bins, dist, {"Omega": 0.1, "psi": 1, "Q": capacity, "R": 0.1, "C": 1.0})
    visited = [n for n in route if n != 0]
    assert route[0] == 0 and route[-1] == 0
    assert len(visited) == len(set(visited))
    assert 3 in visited
    for trip in _trips(route):
        assert trip and sum(bins[i - 1] for i in trip) <= capacity + 1e-6
    tour_cost = sum(dist[a][b] for a, b in zip(route, route[1:], strict=False))
    assert cost == pytest.approx(tour_cost)
    trips = len(_trips(route))
    assert profit == pytest.approx(0.1 * sum(bins[i - 1] for i in visited) - tour_cost - 0.1 * trips, abs=1e-6)


def test_psi_forces_full_bins(monkeypatch):
    """Eq. (17): a bin at or above psi * 100 % is collected even when it does not pay."""
    bins, dist = _instance(10)
    bins = bins.copy()
    bins[6] = 100.0
    route, _, _ = _solve(bins, dist, {"Omega": 0.1, "psi": 1, "Q": 300.0, "R": 0.0001, "C": 1.0}, mandatory=())
    assert 7 in route


@pytest.mark.parametrize("option", [{"depot_inflow": "le"}, {"solver_tuning": True}, {"link_depot_arcs": True}])
def test_optional_switches_keep_the_optimum(option):
    """eq. (15) as '<=', tuning and the depot-arc link change the search, not the optimum here."""
    bins, dist = _instance(12)
    values = {"Omega": 0.1, "psi": 1, "Q": 300.0, "R": 0.1, "C": 1.0}
    base = _solve(bins, dist, values)
    other = _solve(bins, dist, {**values, **option})
    assert other[1] == pytest.approx(base[1], rel=0.011)


def test_arc_cutoff_drops_long_edges():
    bins, dist = _instance(12)
    values = {"Omega": 0.1, "psi": 1, "Q": 300.0, "R": 0.1, "C": 1.0, "max_arc_distance_km": 1e-6}
    # Only zero-length arcs survive: no bin can be reached, so forcing bin 3 is infeasible.
    with pytest.raises(RuntimeError, match="infeasible"):
        _solve(bins, dist, values)
    route, _, _ = _solve(bins, dist, {**values, "relax_forced_on_infeasible": True})
    assert route == [0, 0]
