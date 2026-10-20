"""Regression test: the local-search Top-3 insertion cache never holds stale entries.

Entries used to survive across ``_optimize_internal`` calls, for nodes that became
routed, and for routes that became empty, so unrouted insertions read wrong costs.
"""

import random

import numpy as np
import pytest
from logic.src.configs.policies import HGSConfig
from logic.src.policies.helpers.local_search.local_search_hgs import HGSLocalSearch


def _instance(n, seed, scale):
    rng = np.random.default_rng(seed)
    pts = rng.random((n + 1, 2))
    wastes = {i: float(rng.random() * scale) for i in range(1, n + 1)}
    return pts, wastes


def _assert_cache_fresh(ls):
    routed = set(ls.node_map)
    for node, per_route in ls.top_insertions.items():
        assert node not in routed, f"cache entry kept for routed node {node}"
        for r_idx, top in per_route.items():
            route = ls.routes[r_idx]
            costs = sorted(
                (
                    ls.get_dist(route[p - 1] if p > 0 else 0, node)
                    + ls.get_dist(node, route[p] if p < len(route) else 0)
                    - ls.get_dist(route[p - 1] if p > 0 else 0, route[p] if p < len(route) else 0),
                    p,
                )
                for p in range(len(route) + 1)
            )[:3]
            assert top == pytest.approx(costs), f"stale cache for node {node}, route {r_idx}"


def test_top_insertion_cache_is_never_stale():
    n = 14
    pts, wastes = _instance(n=n, seed=3, scale=0.4)
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    cfg = HGSConfig(local_search_iterations=200, seed=1)
    ls = HGSLocalSearch(dist, wastes, 1.0, 1.0, 1.0, None, cfg)
    rng = random.Random(0)
    for _ in range(12):
        nodes = list(range(1, n + 1))
        rng.shuffle(nodes)
        k = rng.randint(n // 2, n)  # some calls leave bins unrouted, some route all
        chosen, routes, load = nodes[:k], [[]], 0.0
        for v in chosen:
            if load + wastes[v] > 1.0:
                routes.append([])
                load = 0.0
            routes[-1].append(v)
            load += wastes[v]
        ls.routes = routes
        ls._optimize_internal()
        _assert_cache_fresh(ls)
