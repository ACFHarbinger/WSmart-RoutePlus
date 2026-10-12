"""Asymmetric (directed-arc) SWC-TCF model: agreement with the published model and route consistency."""

import numpy as np
import pytest
from logic.src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow import (
    gurobi,
)


def _instance(seed: int, n: int, symmetric: bool):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 10, size=(n + 1, 2))
    dist = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    if not symmetric:
        dist = dist * rng.uniform(0.6, 1.6, size=dist.shape)
    np.fill_diagonal(dist, 0.0)
    bins = rng.uniform(30, 80, size=n)
    return bins, dist.tolist()


def _values(**overrides):
    return {"Omega": 0.1, "psi": 1.0, "Q": 100.0, "R": 1.0, "C": 1.0, **overrides}


def _solve(bins, dist, asymmetric: bool):
    return gurobi._run_gurobi_optimizer(
        bins,
        dist,
        None,
        _values(asymmetric=asymmetric),
        list(range(len(bins) + 1)),
        [],
        number_vehicles=0,
        time_limit=30,
    )


def _trips(route):
    trips, current = [], []
    for node in route[1:]:
        if node == 0:
            if current:
                trips.append(current)
            current = []
        else:
            current.append(node)
    return trips


@pytest.mark.parametrize("seed", [0, 1])
def test_symmetric_matrix_matches_published_model(seed):
    bins, dist = _instance(seed, 5, symmetric=True)
    _, published, _ = _solve(bins, dist, asymmetric=False)
    _, asymmetric, _ = _solve(bins, dist, asymmetric=True)
    assert asymmetric == pytest.approx(published, rel=1e-6, abs=1e-6)


def test_asymmetric_profit_matches_directed_route_cost():
    bins, dist = _instance(0, 5, symmetric=False)
    route, profit, cost = _solve(bins, dist, asymmetric=True)
    trips = _trips(route)
    assert trips, "the instance has positive profit, so a route is expected"
    for trip in trips:
        assert sum(bins[node - 1] for node in trip) <= 100.0 + 1e-6
    arc_cost = sum(dist[a][b] for a, b in zip(route, route[1:], strict=False))
    collected = sum(bins[node - 1] for trip in trips for node in trip)
    assert cost == pytest.approx(arc_cost)
    assert profit == pytest.approx(collected - arc_cost - 0.1 * len(trips), rel=1e-6, abs=1e-6)


def test_default_is_the_undirected_model():
    assert gurobi._options({})["asymmetric"] is False


def test_asymmetric_requires_the_paper_formulation():
    with pytest.raises(ValueError, match="requires formulation: paper"):
        gurobi._run_gurobi_optimizer(
            np.array([50.0]),
            [[0.0, 1.0], [1.0, 0.0]],
            None,
            _values(asymmetric=True, formulation="directed"),
            [1],
            [],
            number_vehicles=0,
        )


def test_asymmetric_rejects_warm_starts():
    with pytest.raises(ValueError, match="warm_start"):
        gurobi._run_gurobi_optimizer(
            np.array([50.0]),
            [[0.0, 1.0], [1.0, 0.0]],
            None,
            _values(asymmetric=True, warm_start=True),
            [1],
            [],
            number_vehicles=0,
        )
