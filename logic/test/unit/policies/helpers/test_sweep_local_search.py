"""SweepLocalSearch: the named operator sweeps on the LocalSearch base."""

import numpy as np
import pytest
from logic.src.policies.helpers.local_search import LocalSearch, SweepLocalSearch

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _instance(seed=0, n=14):
    rng = np.random.default_rng(seed)
    xy = rng.random((n + 1, 2)) * 10
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    wastes = {i: float(rng.uniform(1, 10)) for i in range(1, n + 1)}
    perm = [int(x) for x in rng.permutation(np.arange(1, n + 1))]
    return dist, wastes, [perm[:5], perm[5:10], perm[10:]]


def _cost(dist, routes):
    return sum(dist[a, b] for r in routes if r for a, b in zip([0, *r], [*r, 0], strict=True))


def test_is_a_local_search_with_shared_state():
    dist, wastes, routes = _instance()
    ls = SweepLocalSearch(dist, wastes, 30.0, 1.0, 1.0, seed=1)
    assert isinstance(ls, LocalSearch)
    ls.set_routes(routes)
    assert ls.route_loads == [sum(wastes[i] for i in r) for r in routes]
    assert ls.node_map[routes[1][2]] == (1, 2)
    assert ls.get_routes() == routes and ls.get_routes() is not ls.routes


@pytest.mark.parametrize("op", ["relocate", "swap", "two_opt_intra", "or_opt", "two_opt_star", "swap_star"])
def test_each_sweep_keeps_the_nodes_and_never_lengthens_the_tour(op):
    dist, wastes, routes = _instance(2)
    ls = SweepLocalSearch(dist, wastes, 1e9, 1.0, 1.0, seed=2)
    ls.set_routes(routes)
    cost = _cost(dist, routes)
    for _ in range(50):
        if not getattr(ls, op)():
            break
        new_cost = _cost(dist, ls.routes)
        assert new_cost < cost - 1e-9, f"{op} applied a non-improving move"
        cost = new_cost
    after = ls.get_routes()
    assert sorted(n for r in after for n in r) == sorted(n for r in routes for n in r)
    assert ls.route_loads == [sum(wastes[i] for i in r) for r in after]


def test_relocate_never_moves_a_node_after_itself():
    """Relocating u after u is a hidden swap with its successor; it cycled the old sweep."""
    dist, wastes, routes = _instance(3)
    ls = SweepLocalSearch(dist, wastes, 1e9, 1.0, 1.0, seed=3)
    ls.set_routes(routes)
    seen = {str(ls.routes)}
    while ls.relocate():
        assert str(ls.routes) not in seen
        seen.add(str(ls.routes))


def test_optimize_descends_to_a_local_optimum():
    dist, wastes, routes = _instance(3)
    ls = SweepLocalSearch(dist, wastes, 1e9, 1.0, 1.0, seed=3)
    best = ls.optimize(routes)
    assert _cost(dist, best) < _cost(dist, routes)
    ls.set_routes(best)
    assert not any(getattr(ls, op)() for op in ("relocate", "swap", "two_opt_intra", "two_opt_star"))
