"""k-opt* tail exchange: the reported gain equals the change in real route cost."""

import numpy as np
import pytest
from logic.src.policies.helpers.operators.inter_route_local_search.k_opt_star import (
    _apply_tail_permutation,
    _extract_route_parts,
    _find_best_tail_permutation,
)
from logic.test.unit.policies.matheuristics.test_ils_rvnd_sp_paper import MockLocalSearch


def _route_cost(dist, route):
    path = [0, *route, 0]
    return float(sum(dist[a, b] for a, b in zip(path, path[1:], strict=False)))  # one edge per consecutive pair


def _total(dist, routes):
    return sum(_route_cost(dist, r) for r in routes)


def _random_instance(seed, n=9):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0, 100, size=(n + 1, 2))
    dist = np.sqrt(((points[:, None] - points[None]) ** 2).sum(-1))
    nodes = rng.permutation(np.arange(1, n + 1)).tolist()
    return dist, [nodes[:4], nodes[4:]]


@pytest.mark.parametrize("seed", range(40))
def test_two_route_gain_matches_the_real_cost_change(seed):
    dist, routes = _random_instance(seed)
    ls = MockLocalSearch(routes, dist, capacity=1e9)
    before = _total(dist, routes)

    cuts = [(routes[0][1], 0, 1), (routes[1][1], 1, 1)]
    heads, tails, head_loads, tail_loads, original = _extract_route_parts(ls, cuts)
    gain, perm = _find_best_tail_permutation(ls, cuts, heads, tails, head_loads, tail_loads, original)

    if perm is None:
        assert gain == 0.0
        return
    assert gain > 0.0
    _apply_tail_permutation(ls, cuts, heads, tails, perm)
    assert _total(dist, ls.routes) == pytest.approx(before - gain, abs=1e-9)


def test_improving_moves_occur_in_the_sample():
    improving = 0
    for seed in range(40):
        dist, routes = _random_instance(seed)
        ls = MockLocalSearch(routes, dist, capacity=1e9)
        cuts = [(routes[0][1], 0, 1), (routes[1][1], 1, 1)]
        parts = _extract_route_parts(ls, cuts)
        gain, perm = _find_best_tail_permutation(ls, cuts, *parts[:4], parts[4])
        improving += perm is not None
    assert improving > 0


def test_two_cut_swap_reconnects_u_to_v_successor_and_v_to_u_successor():
    points = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]], dtype=float)
    dist = np.sqrt(((points[:, None] - points[None]) ** 2).sum(-1))
    routes = [[1, 2, 3, 4], [5, 6, 7, 8]]
    ls = MockLocalSearch(routes, dist, capacity=1e9)
    u, v = 2, 6
    cuts = [(u, 0, 1), (v, 1, 1)]
    heads, tails, _, _, original = _extract_route_parts(ls, cuts)

    _apply_tail_permutation(ls, cuts, heads, tails, (1, 0))

    # u keeps its head and now precedes v's old tail (starting at 7); v precedes u's old tail (starting at 3)
    assert ls.routes[0] == [1, 2, 7, 8]
    assert ls.routes[1] == [5, 6, 3, 4]
    old_cut = dist[u, 3] + dist[v, 7]
    new_cut = dist[u, 7] + dist[v, 3]
    assert original == pytest.approx(old_cut)
    assert _total(dist, ls.routes) == pytest.approx(_total(dist, [[1, 2, 3, 4], [5, 6, 7, 8]]) - old_cut + new_cut)
