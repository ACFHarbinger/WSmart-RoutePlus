"""SANS: an over-capacity day is served in several trips; the overflow rule (perc_bins_can_overflow) is enforced."""

import itertools
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pytest
import yaml
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.dispatcher import (
    split_into_trips,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.heuristics import sans


def _trips(tour: List[int]) -> List[List[int]]:
    out: List[List[int]] = [[]]
    for node in tour[1:]:
        if node == 0:
            out.append([])
        else:
            out[-1].append(node)
    return [t for t in out if t]


def _length(d: np.ndarray, tour: List[int]) -> float:
    return float(sum(d[a][b] for a, b in zip(tour, tour[1:], strict=False)))


def test_a_tour_within_capacity_is_unchanged() -> None:
    d = np.ones((4, 4)) - np.eye(4)
    tour = [0, 1, 2, 3, 0]
    assert split_into_trips(tour, {1: 30.0, 2: 30.0, 3: 30.0}, 100.0, d) is tour


@pytest.mark.parametrize("seed", range(5))
def test_split_trips_fit_the_vehicle_keep_the_order_and_are_optimal(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 7
    xy = rng.uniform(0, 10, (n + 1, 2))
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    stocks: Dict[int, float] = {i: float(rng.uniform(40, 95)) for i in range(1, n + 1)}
    order = [int(b) for b in rng.permutation(np.arange(1, n + 1))]
    tour = split_into_trips([0, *order, 0], stocks, 150.0, d)
    assert [b for b in tour if b != 0] == order  # same bins, same order
    assert all(sum(stocks[b] for b in t) <= 150.0 for t in _trips(tour))
    # the cheapest of all ways to cut the order into capacity-feasible trips
    best = float("inf")
    for cuts in itertools.product([False, True], repeat=n - 1):
        trips, cur = [], [order[0]]
        for b, cut in zip(order[1:], cuts, strict=True):
            if cut:
                trips.append(cur)
                cur = []
            cur.append(b)
        trips.append(cur)
        if all(sum(stocks[b] for b in t) <= 150.0 for t in trips):
            best = min(best, _length(d, [0, *[x for t in trips for x in (*t, 0)]]))
    assert _length(d, tour) == pytest.approx(best)


def test_a_bin_larger_than_the_vehicle_gets_its_own_trip() -> None:
    d = np.ones((4, 4)) - np.eye(4)
    tour = split_into_trips([0, 1, 2, 3, 0], {1: 30.0, 2: 250.0, 3: 30.0}, 100.0, d)
    assert [2] in _trips(tour) and sorted(b for b in tour if b) == [1, 2, 3]


def _sans(perc: float, seed: int = 0) -> Tuple[Set[int], Set[int]]:
    rng = np.random.default_rng(seed)
    n = 20
    xy = rng.uniform(0, 10, (n + 1, 2))
    d = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    stock = rng.uniform(5, 60, n)
    rate = np.full(n, 2.0)
    rate[:6] = 60.0  # bins 1-6 overflow by tomorrow (stock + rate >= 100) when their stock is >= 40
    data = pd.DataFrame({"#bin": np.arange(n + 1), "Stock": [0.0, *stock], "Accum_Rate": [0.0, *rate]})
    routes, *_ = sans.improved_simulated_annealing(
        [[0, 7, 0]], d, 600, {i: i for i in range(n + 1)}, data, 7368.0, 75, 0.01, 0.95, 200, 1e-4, 100.0, 1.0, 1.0,
        [7], perc_bins_can_overflow=perc, rng=random.Random(seed), max_iterations=3000)
    overflowing = {i + 1 for i in range(n) if stock[i] + rate[i] >= 100}
    return overflowing, {b for r in routes for b in r}


def test_overflow_rule_requires_the_overflowing_bins() -> None:
    overflowing, visited = _sans(0.0)
    assert overflowing  # the instance has some
    assert overflowing <= visited  # 0: every one of them, although they barely pay (revenue 1e-4 per percent)
    _, loose = _sans(1.0)
    assert not overflowing <= loose  # 1.0: no requirement, and unprofitable bins are left
    _, half = _sans(len(overflowing) / 2 / 20)  # may leave half of them
    assert len(overflowing & half) >= len(overflowing) - len(overflowing) // 2


def test_paper_yaml_sets_the_rule_off() -> None:
    with open("logic/configs/policies/routing/meta_heuristics/policy_sans.yaml") as fh:
        raw = yaml.safe_load(fh)
    assert raw["sans"]["new"]["perc_bins_can_overflow"] == 1.0
