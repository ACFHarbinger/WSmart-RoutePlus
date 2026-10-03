"""The og uncross walk must stop when the caller's deadline has passed.

A long crossed route used to keep swapping until every crossing was gone,
ignoring the time limit find_solutions was given. The deadline is carried
on a copy of values as '_uncross_deadline' (perf_counter seconds).
"""

import math
import time

import numpy as np
import pandas as pd
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.common.routes import (
    uncross_arcs_in_routes,
)
from logic.src.policies.route_construction.meta_heuristics.simulated_annealing_neighborhood_search.refinement import (
    route_search,
)


def _frame(n, points):
    distance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            distance[i, j] = math.hypot(dx, dy)
    data = pd.DataFrame({"Stock": np.ones(n), "Accum_Rate": np.zeros(n), "#bin": np.arange(n)})
    values = {
        "E": 1.0,
        "B": 1.0,
        "R": 1.0,
        "C": 1.0,
        "vehicle_capacity": 1.0e6,
        "shift_duration": 1.0e9,
        "perc_bins_can_overflow": 1.0,
    }
    return distance, data, values


def _circle_instance(n_customers):
    n = n_customers + 1
    points = []
    for i in range(n):
        angle = 2.0 * math.pi * i / n_customers
        points.append([math.cos(angle), math.sin(angle)])
    order = list(range(1, n))
    # Interleave the two halves so consecutive arcs cross.
    route = [0] + order[::2] + order[1::2] + [0]
    distance, data, values = _frame(n, points)
    return [route], points, distance, data, values


def _bowtie_instance():
    """Depot plus one proper crossing: arc 1-2 crosses arc 3-4."""
    points = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.2], [1.0, 1.0]]
    route = [0, 1, 2, 3, 4, 0]
    distance, data, values = _frame(len(points), points)
    return [route], points, distance, data, values


def _uncross(solution, points, distance, data, values):
    return uncross_arcs_in_routes(
        solution,
        0.0,
        0.0,
        0.0,
        0.0,
        data,
        points,
        distance,
        values,
    )


def test_uncross_returns_before_a_tight_deadline():
    solution, points, distance, data, values = _circle_instance(40)
    values = dict(values)
    values["_uncross_deadline"] = time.perf_counter() + 0.02
    started = time.perf_counter()
    routes, _, _ = _uncross(solution, points, distance, data, values)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.15, elapsed
    assert routes[0][0] == 0 and routes[0][-1] == 0
    assert sorted(routes[0][1:-1]) == list(range(1, 41))


def test_expired_deadline_leaves_the_crossed_route_unchanged():
    solution, points, distance, data, values = _bowtie_instance()
    original = list(solution[0])
    values = dict(values)
    values["_uncross_deadline"] = time.perf_counter() - 1.0
    routes, _, _ = _uncross(solution, points, distance, data, values)
    assert routes[0] == original
    assert solution[0] == original


def test_uncross_without_a_deadline_still_removes_a_crossing():
    solution, points, distance, data, values = _bowtie_instance()
    original = list(solution[0])
    routes, _, _ = _uncross(solution, points, distance, data, values)
    assert routes[0] != original
    assert routes[0][0] == 0 and routes[0][-1] == 0
    assert sorted(routes[0][1:-1]) == [1, 2, 3, 4]


def test_find_solutions_stamps_the_deadline_on_a_copy():
    captured = {}

    def fake_initial(*_args, **_kwargs):
        return [[0, 1, 0]]

    def fake_uncross(sol, *_args):
        captured["uncross_values"] = _args[-1]
        return sol, 0.0, 0.0

    def fake_anneal(*_args, **_kwargs):
        return [[0, 1, 0]], []

    def fake_refine(sol, *_args, **_kwargs):
        captured["refine_values"] = _args[-1]
        return sol, 0.0

    def fake_rebalance(*_args, **_kwargs):
        return [[0, 1, 0]], 0.0, []

    originals = {
        "find_initial_solution": route_search.find_initial_solution,
        "uncross_arcs_in_routes": route_search.uncross_arcs_in_routes,
        "run_annealing_loop": route_search.run_annealing_loop,
        "refine_solution": route_search.refine_solution,
        "rebalance_solution": route_search.rebalance_solution,
    }
    route_search.find_initial_solution = fake_initial
    route_search.uncross_arcs_in_routes = fake_uncross
    route_search.run_annealing_loop = fake_anneal
    route_search.refine_solution = fake_refine
    route_search.rebalance_solution = fake_rebalance

    caller = {"E": 1.0, "B": 1.0, "vehicle_capacity": 1.0}
    snapshot = dict(caller)
    tic = time.perf_counter()
    try:
        route_search.find_solutions(
            None,
            None,
            None,
            [1, 0, 0, 0, 0, 0, 0],
            [],
            caller,
            1,
            [],
            0.5,
            None,
            None,
        )
        assert caller == snapshot
        assert "_uncross_deadline" not in caller
        stamped = captured["uncross_values"]
        assert stamped is not caller
        assert stamped is captured["refine_values"]
        deadline = stamped["_uncross_deadline"]
        assert tic + 0.5 - 1e-3 <= deadline <= time.perf_counter() + 0.5 + 1e-3
    finally:
        for name, fn in originals.items():
            setattr(route_search, name, fn)
