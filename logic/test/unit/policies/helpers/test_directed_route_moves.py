"""Route moves on asymmetric distances: every applied move must shorten the directed route."""

import random

import numpy as np
import pytest
from logic.src.policies.helpers.local_search import SweepLocalSearch
from logic.src.policies.helpers.operators.intra_route_local_search.k_opt import (
    directed_two_opt_route,
    move_kopt_intra,
    reversal_delta,
    two_opt_route,
)
from logic.src.policies.routing.other_algorithms.travelling_salesman_problem.tsp import find_route

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _asymmetric(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n + 1, 2)) * 10
    d = np.linalg.norm(xy[:, None] - xy[None], axis=-1) * rng.uniform(0.6, 1.4, (n + 1, n + 1))
    np.fill_diagonal(d, 0.0)
    return d


def _length(d, bins):
    path = [0, *bins, 0]
    return sum(d[a, b] for a, b in zip(path, path[1:], strict=False))


def _ring(n):
    """Directed ring 0 -> 1 -> ... -> n -> 0 costs n + 1; every other arc costs 100."""
    d = np.full((n + 1, n + 1), 100.0)
    np.fill_diagonal(d, 0.0)
    for k in range(n + 1):
        d[k, (k + 1) % (n + 1)] = 1.0
    return d


def test_reversal_delta_prices_the_interior_backwards():
    d = _ring(4)
    assert reversal_delta(d, [1, 2, 3]) == pytest.approx(2 * 100 - 2 * 1)
    assert reversal_delta(np.ones((4, 4)), [1, 2, 3]) == 0.0


@pytest.mark.parametrize("n", [5, 12, 25])
def test_find_route_drives_a_directed_ring_forward(n):
    tour = find_route(_ring(n), list(range(1, n + 1)), time_limit=0.2)
    assert _length(_ring(n), tour[1:-1]) == pytest.approx(n + 1)


@pytest.mark.parametrize("seed", range(5))
def test_directed_two_opt_route_never_lengthens(seed):
    d = _asymmetric(15, seed)
    route = list(np.random.default_rng(seed).permutation(np.arange(1, 16)))
    assert _length(d, directed_two_opt_route(route, d)) <= _length(d, route) + 1e-9
    assert _length(d, two_opt_route(route, d)) <= _length(d, route) + 1e-9


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("op", ["two_opt_intra", "three_opt_intra", "four_opt_intra"])
def test_sweep_moves_shorten_the_directed_route(seed, op):
    d = _asymmetric(14, seed)
    perm = [int(x) for x in np.random.default_rng(seed).permutation(np.arange(1, 15))]
    ls = SweepLocalSearch(d, {i: 1.0 for i in range(1, 15)}, 1e9, 1.0, 1.0, seed=seed)
    ls.set_routes([perm[:7], perm[7:]])
    cost = sum(_length(d, r) for r in ls.routes)
    for _ in range(60):
        if not getattr(ls, op)():
            break
        new_cost = sum(_length(d, r) for r in ls.routes)
        assert new_cost < cost - 1e-9
        cost = new_cost


def test_two_opt_sweep_can_reverse_a_whole_route():
    d = _ring(2)
    ls = SweepLocalSearch(d, {1: 1.0, 2: 1.0}, 10.0, 1.0, 1.0, seed=0)
    ls.set_routes([[2, 1]])
    assert ls.two_opt_intra()
    assert ls.routes == [[1, 2]]


@pytest.mark.parametrize("k", [3, 4])
def test_kopt_applied_moves_shorten_the_directed_route(k):
    d = _asymmetric(12, 7)
    ls = SweepLocalSearch(d, {i: 1.0 for i in range(1, 13)}, 1e9, 1.0, 1.0, seed=7)
    ls.set_routes([list(range(1, 13))])
    rng = random.Random(1)
    for _ in range(300):
        p_u, p_v = sorted(rng.sample(range(12), 2))
        before = _length(d, ls.routes[0])
        if move_kopt_intra(ls, ls.routes[0][p_u], ls.routes[0][p_v], 0, p_u, 0, p_v, k=k, rng=rng):
            assert _length(d, ls.routes[0]) < before - 1e-9
