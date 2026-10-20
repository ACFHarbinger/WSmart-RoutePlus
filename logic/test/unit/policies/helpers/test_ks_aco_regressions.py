"""Regression tests for three K-sparse ACO bugs found while matching the vectorised experts.

1. K-sparse ACO with ``local_search=True`` crashed (``ACOLocalSearch`` had no distance matrix).
2. K-sparse ACO raised ``ZeroDivisionError`` when its best solution had zero cost.
3. ``KSparseACOSolver._nearest_neighbor_cost`` looped forever on a bin larger than the capacity.
"""

import threading

import numpy as np
from logic.src.configs.policies import KSparseACOConfig
from logic.src.policies.routing.meta_heuristics.ant_colony_optimization_k_sparse.solver import KSparseACOSolver


def _instance(n=15, seed=0, scale=0.5):
    rng = np.random.default_rng(seed)
    pts = rng.random((n + 1, 2))
    wastes = {i: float(rng.random() * scale) for i in range(1, n + 1)}
    return pts, wastes


def test_ks_aco_local_search_runs():
    pts, wastes = _instance()
    cfg = KSparseACOConfig(n_ants=4, max_iterations=3, local_search=True, seed=1)
    routes, profit, cost = KSparseACOSolver(pts, wastes, 1.0, 5.0, 1.0, cfg).solve()
    visited = [n for r in routes for n in r]
    assert len(visited) == len(set(visited))
    assert cost >= 0.0 and np.isfinite(profit)


def test_ks_aco_zero_cost_best_does_not_divide_by_zero():
    # Revenue far below any round trip: the ants build empty solutions (cost 0).
    pts, wastes = _instance(scale=1e-6)
    cfg = KSparseACOConfig(n_ants=3, max_iterations=3, local_search=False, seed=1)
    routes, profit, cost = KSparseACOSolver(pts, wastes, 1.0, 1.0, 1.0, cfg).solve()
    assert cost == 0.0 and profit == 0.0
    assert not any(routes)


def _within(seconds, fn):
    """Run ``fn`` in a daemon thread; fail instead of hanging if it does not return."""
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("value", fn()), daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "did not terminate"
    return result["value"]


def test_nearest_neighbor_cost_skips_bins_over_capacity():
    pts, wastes = _instance(n=8)
    wastes[3] = 2.0  # never fits a vehicle of capacity 1
    cfg = KSparseACOConfig(n_ants=2, max_iterations=1, local_search=False, tau_0=1.0, seed=1)
    solver = KSparseACOSolver(pts, wastes, 1.0, 1.0, 1.0, cfg)
    assert np.isfinite(_within(30, solver._nearest_neighbor_cost))
    # tau_0 = None goes through the same computation in __init__.
    cfg2 = KSparseACOConfig(n_ants=2, max_iterations=1, local_search=False, seed=1)
    assert np.isfinite(_within(30, lambda: KSparseACOSolver(pts, wastes, 1.0, 1.0, 1.0, cfg2)).tau_0)
