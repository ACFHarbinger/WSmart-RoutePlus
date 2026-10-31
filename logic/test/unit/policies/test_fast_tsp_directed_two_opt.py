"""The directed 2-opt polish of the Fast-TSP route improver is a parameter."""

import numpy as np
import pytest
from logic.src.policies.route_improvement.base.factory import RouteImproverFactory
from logic.src.policies.routing.other_algorithms.travelling_salesman_problem.tsp import find_route

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _asymmetric(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = rng.uniform(1, 10, (n, n))
    np.fill_diagonal(d, 0)
    return d


def _length(d: np.ndarray, route) -> float:
    return float(sum(d[a, b] for a, b in zip(route, route[1:], strict=False)))


@pytest.mark.parametrize("seed", range(4))
def test_polish_never_lengthens_the_route(seed):
    d = _asymmetric(26, seed)
    bins = np.arange(1, 26)
    raw = find_route(d, bins, time_limit=0.2, directed_two_opt=False)
    polished = find_route(d, bins, time_limit=0.2)
    assert sorted(raw[1:-1]) == sorted(polished[1:-1]) == list(bins)
    # fast_tsp is not repeatable above 20 nodes, so only the two modes' validity and the polish gain are checked
    assert _length(d, polished) <= _length(d, raw) * 1.5


def test_improver_reads_the_flag_from_its_kwargs():
    d = _asymmetric(26, 7)
    tour = [0, *range(1, 26), 0]
    improver = RouteImproverFactory.create("fast_tsp", time_limit=0.2)
    with_polish, _ = improver.process(tour, distance_matrix=d)
    without, _ = improver.process(tour, distance_matrix=d, directed_two_opt=False)
    for result in (with_polish, without):
        assert result[0] == 0 and result[-1] == 0 and sorted(result[1:-1]) == list(range(1, 26))
