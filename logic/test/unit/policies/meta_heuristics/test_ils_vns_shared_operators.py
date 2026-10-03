"""Tests for the shared destroy-and-repair LLHs used by ILS and VNS.

The ILS and VNS solvers delegate their low-level heuristics (LLHs), shaking
neighborhoods, objective helpers, and initial-solution construction to
``logic.src.policies.helpers.operators.search_heuristics.destroy_repair_llh``.
These tests pin that wiring and prove the shared composites are call-identical
to the legacy inline bodies that used to live in each solver.

Note on determinism: ``worst_removal``/``worst_profit_removal`` fall back to
``np.random.default_rng()`` (OS entropy) when called without an rng — a
pre-existing property of the legacy code, preserved verbatim by the shared
composites. Tests that compare against legacy behaviour therefore patch
``np.random.default_rng`` with a fixed seed so both sides draw identically.
"""

import inspect
import random

import numpy as np
import pytest
from logic.src.policies.helpers.operators.destroy_ruin import (
    cluster_removal,
    random_removal,
    worst_profit_removal,
    worst_removal,
)
from logic.src.policies.helpers.operators.recreate_repair import (
    greedy_insertion,
    greedy_profit_insertion,
    regret_2_insertion,
    regret_2_profit_insertion,
)
from logic.src.policies.helpers.operators.search_heuristics.destroy_repair_llh import (
    build_greedy_initial_routes,
    llh_cluster_greedy,
    llh_random_greedy,
    llh_random_regret_2,
    llh_worst_greedy,
    llh_worst_regret_2,
    routes_net_profit,
    routes_total_distance,
)
from logic.src.policies.route_construction.meta_heuristics.iterated_local_search.params import ILSParams
from logic.src.policies.route_construction.meta_heuristics.iterated_local_search.solver import ILSSolver
from logic.src.policies.route_construction.meta_heuristics.variable_neighborhood_search.params import VNSParams
from logic.src.policies.route_construction.meta_heuristics.variable_neighborhood_search.solver import VNSSolver

CAPACITY = 60.0
R_REVENUE = 2.0
C_COST = 1.0
MANDATORY = [3, 7]


def make_instance(n: int = 12) -> tuple:
    """Fixed 12-node instance."""
    rng = np.random.default_rng(5)
    coords = rng.uniform(0, 10, size=(n + 1, 2))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    wrng = np.random.default_rng(9)
    wastes = {i: float(wrng.uniform(5, 25)) for i in range(1, n + 1)}
    return dist, wastes


def make_solver_instance(n: int = 15) -> tuple:
    """Fixed 15-node instance for solver-level tests."""
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 10, size=(n + 1, 2))
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    wrng = np.random.default_rng(11)
    wastes = {i: float(wrng.uniform(5, 25)) for i in range(1, n + 1)}
    return dist, wastes


# ----------------------------------------------------------------------
# Legacy inline bodies, copied from the pre-refactor solvers.
# ----------------------------------------------------------------------


def legacy_llh0(routes, n, dist, wastes, rng):
    partial, removed = random_removal(routes, n, rng)
    return greedy_insertion(
        partial, removed, dist, wastes, CAPACITY, mandatory_nodes=MANDATORY, expand_pool=True
    )


def legacy_llh1(routes, n, dist, wastes, rng):
    partial, removed = worst_removal(routes, n, dist)
    return regret_2_insertion(
        partial, removed, dist, wastes, CAPACITY, mandatory_nodes=MANDATORY, expand_pool=True
    )


def legacy_llh2(routes, n, dist, wastes, rng, nodes):
    partial, removed = cluster_removal(routes, n, dist, nodes, rng)
    return greedy_insertion(
        partial, removed, dist, wastes, CAPACITY, mandatory_nodes=MANDATORY, expand_pool=True
    )


def legacy_llh3(routes, n, dist, wastes, rng):
    partial, removed = worst_removal(routes, n, dist)
    return greedy_insertion(
        partial, removed, dist, wastes, CAPACITY, mandatory_nodes=MANDATORY, expand_pool=True
    )


def legacy_llh4(routes, n, dist, wastes, rng):
    partial, removed = random_removal(routes, n, rng)
    return regret_2_insertion(
        partial, removed, dist, wastes, CAPACITY, mandatory_nodes=MANDATORY, expand_pool=True
    )


def legacy_llh1_profit(routes, n, dist, wastes, rng):
    partial, removed = worst_profit_removal(routes, n, dist, wastes, R_REVENUE, C_COST)
    return regret_2_profit_insertion(
        partial, removed, dist, wastes, CAPACITY, R_REVENUE, C_COST, MANDATORY, True
    )


def legacy_llh3_profit(routes, n, dist, wastes, rng):
    partial, removed = worst_profit_removal(routes, n, dist, wastes, R_REVENUE, C_COST)
    return greedy_profit_insertion(
        partial, removed, dist, wastes, CAPACITY, R_REVENUE, C_COST, MANDATORY, True
    )


def legacy_llh0_profit(routes, n, dist, wastes, rng):
    partial, removed = random_removal(routes, n, rng)
    return greedy_profit_insertion(partial, removed, dist, wastes, CAPACITY, R_REVENUE, C_COST, MANDATORY, True)


def legacy_llh2_profit(routes, n, dist, wastes, rng, nodes):
    partial, removed = cluster_removal(routes, n, dist, nodes, rng)
    return greedy_profit_insertion(partial, removed, dist, wastes, CAPACITY, R_REVENUE, C_COST, MANDATORY, True)


def legacy_llh4_profit(routes, n, dist, wastes, rng):
    partial, removed = random_removal(routes, n, rng)
    return regret_2_profit_insertion(partial, removed, dist, wastes, CAPACITY, R_REVENUE, C_COST, MANDATORY, True)


# (name, legacy_fn, shared_factory) — shared_factory(pa) returns the composite call
def _shared(name):
    def factory(pa):
        if name == "random_greedy":
            return lambda r, n, d, w, rng: llh_random_greedy(r, n, d, w, CAPACITY, R_REVENUE, C_COST, MANDATORY, True, pa, rng)
        if name == "worst_regret_2":
            return lambda r, n, d, w, rng: llh_worst_regret_2(r, n, d, w, CAPACITY, R_REVENUE, C_COST, MANDATORY, True, pa, rng)
        if name == "cluster_greedy":
            return lambda r, n, d, w, rng: llh_cluster_greedy(r, n, d, w, CAPACITY, R_REVENUE, C_COST, MANDATORY, True, pa, rng, nodes=list(range(1, len(d))))
        if name == "worst_greedy":
            return lambda r, n, d, w, rng: llh_worst_greedy(r, n, d, w, CAPACITY, R_REVENUE, C_COST, MANDATORY, True, pa, rng)
        return lambda r, n, d, w, rng: llh_random_regret_2(r, n, d, w, CAPACITY, R_REVENUE, C_COST, MANDATORY, True, pa, rng)

    return factory


PLAIN_PAIRS = [
    ("random_greedy", legacy_llh0, _shared("random_greedy")),
    ("worst_regret_2", legacy_llh1, _shared("worst_regret_2")),
    ("cluster_greedy", lambda r, n, d, w, rng: legacy_llh2(r, n, d, w, rng, list(range(1, len(d)))), _shared("cluster_greedy")),
    ("worst_greedy", legacy_llh3, _shared("worst_greedy")),
    ("random_regret_2", legacy_llh4, _shared("random_regret_2")),
]

PROFIT_PAIRS = [
    ("random_greedy", legacy_llh0_profit, _shared("random_greedy")),
    ("worst_regret_2", legacy_llh1_profit, _shared("worst_regret_2")),
    ("cluster_greedy", lambda r, n, d, w, rng: legacy_llh2_profit(r, n, d, w, rng, list(range(1, len(d)))), _shared("cluster_greedy")),
    ("worst_greedy", legacy_llh3_profit, _shared("worst_greedy")),
    ("random_regret_2", legacy_llh4_profit, _shared("random_regret_2")),
]

BASE_ROUTES = [[1, 4, 6], [2, 8, 10], [3, 5, 9, 12], [7, 11]]


@pytest.mark.parametrize("profit_aware", [False, True], ids=["plain", "profit"])
@pytest.mark.parametrize("n_remove", [1, 2, 3])
@pytest.mark.parametrize("pair_idx", range(5), ids=[p[0] for p in PLAIN_PAIRS])
def test_shared_composite_matches_legacy_body(pair_idx, n_remove, profit_aware, monkeypatch):
    """Each shared LLH must reproduce the legacy inline body bit-for-bit."""
    dist, wastes = make_instance()
    pairs = PROFIT_PAIRS if profit_aware else PLAIN_PAIRS
    name, legacy_fn, shared_factory = pairs[pair_idx]

    real_default_rng = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: real_default_rng(31337 + n_remove))

    legacy_out = legacy_fn([r[:] for r in BASE_ROUTES], n_remove, dist, wastes, random.Random(1000 + n_remove))
    shared_out = shared_factory(profit_aware)([r[:] for r in BASE_ROUTES], n_remove, dist, wastes, random.Random(1000 + n_remove))
    assert shared_out == legacy_out, f"{name} diverged from the legacy body"


def test_solvers_delegate_to_shared_operators():
    """The solvers must not carry their own operator bodies anymore."""
    expected = {
        "_llh0": "llh_random_greedy",
        "_llh1": "llh_worst_regret_2",
        "_llh2": "llh_cluster_greedy",
        "_llh3": "llh_worst_greedy",
        "_llh4": "llh_random_regret_2",
    }
    for cls in (ILSSolver, VNSSolver):
        for method_name, fn_name in expected.items():
            src = inspect.getsource(getattr(cls, method_name))
            assert fn_name in src, f"{cls.__name__}.{method_name} does not delegate to {fn_name}"
            for primitive in ("random_removal(", "worst_removal(", "greedy_insertion(", "regret_2_insertion("):
                assert primitive not in src, f"{cls.__name__}.{method_name} still calls {primitive}"
    for shake_name, fn_name in {
        "_shake_n1": "llh_random_greedy",
        "_shake_n2": "llh_random_greedy",
        "_shake_n3": "llh_worst_regret_2",
        "_shake_n4": "llh_cluster_greedy",
        "_shake_n5": "llh_random_regret_2",
    }.items():
        src = inspect.getsource(getattr(VNSSolver, shake_name))
        assert fn_name in src, f"VNSSolver.{shake_name} does not delegate to {fn_name}"


def test_shared_objective_helpers_match_legacy_methods():
    """routes_total_distance / routes_net_profit must match the old _cost/_evaluate."""
    dist, wastes = make_solver_instance()
    routes = [[8, 4, 1, 11, 15], [7, 5, 10, 6], [9, 13], [14, 12, 3, 2]]

    expected_cost = 0.0
    for route in routes:
        expected_cost += dist[0][route[0]]
        for k in range(len(route) - 1):
            expected_cost += dist[route[k]][route[k + 1]]
        expected_cost += dist[route[-1]][0]
    assert routes_total_distance(routes, dist) == pytest.approx(expected_cost)

    expected_profit = sum(wastes.get(n, 0.0) * R_REVENUE for r in routes for n in r) - expected_cost * C_COST
    assert routes_net_profit(routes, dist, wastes, R_REVENUE, C_COST) == pytest.approx(expected_profit)
    assert routes_net_profit([], dist, wastes, R_REVENUE, C_COST) == 0.0


def test_composites_reinsert_removed_nodes():
    """With expand_pool=False, no mandatory nodes, and ample capacity, node multiset is conserved."""
    dist, wastes = make_instance()
    routes = [r[:] for r in BASE_ROUTES]
    all_nodes = sorted(n for r in routes for n in r)
    big_capacity = 1e9
    for fn in (
        lambda r, n, rng: llh_random_greedy(r, n, dist, wastes, big_capacity, R_REVENUE, C_COST, [], False, False, rng),
        lambda r, n, rng: llh_random_regret_2(r, n, dist, wastes, big_capacity, R_REVENUE, C_COST, [], False, False, rng),
    ):
        out = fn([r[:] for r in routes], 2, random.Random(7))
        assert sorted(n for r in out for n in r) == all_nodes


def test_build_greedy_initial_routes_respects_capacity():
    dist, wastes = make_instance()
    routes = build_greedy_initial_routes(dist, wastes, CAPACITY, R_REVENUE, C_COST, MANDATORY, random.Random(3))
    assert all(sum(wastes.get(n, 0.0) for n in route) <= CAPACITY + 1e-9 for route in routes)
    visited = {n for r in routes for n in r}
    assert set(MANDATORY) <= visited


# Pinned solver-level results with np.random.default_rng patched to a fixed
# seed (see module docstring). Identical to the pre-refactor solvers.
PATCH_SEED = 20261002
EXPECTED_SOLVES = {
    ("ils", False): {"routes": [[8, 4, 1, 11, 15, 14, 7], [5, 10, 2, 13], [9], [3, 12, 6]], "profit": 331.39289446796795, "cost": 61.05038941451336},
    ("ils", True): {"routes": [[8, 4, 1, 11, 15, 7], [2, 3, 12], [6, 10, 5, 14], [13, 9]], "profit": 332.22832613423515, "cost": 60.21495774824608},
    ("vns", False): {"routes": [[4, 1, 11, 15, 10, 5], [6, 12, 3], [8, 14, 7, 2, 13], [9]], "profit": 333.02969139014965, "cost": 59.41359249233165},
    ("vns", True): {"routes": [[13, 9], [6, 10, 5, 7], [8, 4, 1, 11, 15, 14], [2, 3, 12]], "profit": 332.9270898939912, "cost": 59.51619398849018},
}


@pytest.mark.parametrize("profit_aware", [False, True], ids=["plain", "profit"])
@pytest.mark.parametrize("solver_name", ["ils", "vns"])
def test_solver_results_unchanged_after_refactor(solver_name, profit_aware, monkeypatch):
    """Full solver runs (entropy patched) must match the pre-refactor pinned values."""
    dist, wastes = make_solver_instance()
    real_default_rng = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: real_default_rng(PATCH_SEED))

    if solver_name == "ils":
        params = ILSParams(
            n_restarts=40, inner_iterations=25, time_limit=0, seed=123,
            vrpp=True, profit_aware_operators=profit_aware,
        )
        solver = ILSSolver(dist, wastes, CAPACITY, R_REVENUE, C_COST, params, MANDATORY)
    else:
        params = VNSParams(
            k_max=5, max_iterations=60, local_search_iterations=30, time_limit=0, seed=123,
            vrpp=True, profit_aware_operators=profit_aware,
        )
        solver = VNSSolver(dist, wastes, CAPACITY, R_REVENUE, C_COST, params, MANDATORY)

    routes, profit, cost = solver.solve()
    expected = EXPECTED_SOLVES[(solver_name, profit_aware)]
    assert routes == expected["routes"]
    assert profit == pytest.approx(expected["profit"], rel=0, abs=1e-12)
    assert cost == pytest.approx(expected["cost"], rel=0, abs=1e-12)


def test_worst_removal_entropy_fallback_is_preexisting(monkeypatch):
    """Documents the pre-existing non-determinism source preserved by the refactor."""
    dist, wastes = make_instance()
    real_default_rng = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: real_default_rng(999))
    first = llh_worst_greedy([r[:] for r in BASE_ROUTES], 2, dist, wastes, CAPACITY, R_REVENUE, C_COST, MANDATORY, True, False, random.Random(1))
    monkeypatch.setattr(np.random, "default_rng", lambda seed=None: real_default_rng(12345))
    second = llh_worst_greedy([r[:] for r in BASE_ROUTES], 2, dist, wastes, CAPACITY, R_REVENUE, C_COST, MANDATORY, True, False, random.Random(1))
    # Different entropy seeds may give different (equally valid) removals; both must be
    # complete route sets over the same node pool.
    assert sorted(n for r in first for n in r) == sorted(n for r in second for n in r)
