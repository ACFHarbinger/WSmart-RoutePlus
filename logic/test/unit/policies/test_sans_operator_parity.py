"""SANS and LAC operators reproduce their pre-refactor outputs and random streams.

The operators moved from ``simulated_annealing_neighborhood_search`` to
``helpers/operators`` (2026-10-18), and several were merged behind parameters.
Each entry of ``OPS`` is one call form a SANS or LAC call site uses, keyed by the
pre-refactor operator it replaced. For every seed the test records the operator's
result and the generator's next draw; the hashes were taken from the
pre-refactor tree (commit d846e0fa7), so any change in behaviour or in the
number of random draws fails here.
"""

import copy
import hashlib
import json
import random

import numpy as np
import pandas as pd
import pytest
from logic.src.policies.helpers.operators.destroy_ruin import bin_pool as DB
from logic.src.policies.helpers.operators.evolutionary_mutation.random_2opt import get_2opt_neighbors
from logic.src.policies.helpers.operators.improvement_descent.relocate_descent import relocate_descent
from logic.src.policies.helpers.operators.perturbation_shaking import sans as PS
from logic.src.policies.helpers.operators.recreate_repair import bin_pool as RB
from logic.src.policies.helpers.operators.solution_initialization.nearest_neighbor_si import organize_route
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common import routes as R
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.common.objectives import (
    compute_profit,
    compute_real_profit,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.heuristics import (
    sans_operators as SO,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.refinement.rebalancing import (
    rebalance_solution,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.refinement.refinement import (
    refine_solution,
)
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.search import local_search


def _fns(c):
    def pf(routes):
        return compute_profit(routes, 1.0, 1.0, 1.0, 1.0, c["data"], c["dist"], c["values"])

    def rf(routes):
        return compute_real_profit(routes, 1.0, c["data"], c["dist"], c["values"])

    return pf, rf


pytestmark = [pytest.mark.unit]

OPERATOR_SEEDS = range(60)
DRIVER_SEEDS = range(20)

PINNED = {
    "move_1_route": "46a823ee28ce2275",
    "move_2_routes": "7b73b91f39ed1ca0",
    "move_n_route_random": "8346ffac8bc01906",
    "move_n_route_consecutive": "374ea0d7f2590b0e",
    "move_n_2_routes_random": "35832596019538d9",
    "move_n_2_routes_consecutive": "d1ea4f718b9112aa",
    "swap_1_route": "bcc52847d3b365e5",
    "swap_2_routes": "94f5934685fb62a1",
    "swap_n_route_random": "73638fee335cd081",
    "swap_n_route_consecutive": "8ee3f837884001d1",
    "swap_n_2_routes_random": "f892a33686ee666d",
    "swap_n_2_routes_consecutive": "8fe341b34a00af6b",
    "move_n_route_random_3": "53863af6f5b1b34e",
    "swap_n_2_routes_consecutive_3": "1eac6e3416e82214",
    "remove_bin": "b903c98edfbe70ed",
    "remove_n_bins_random": "7053ecf14478b685",
    "remove_n_bins_consecutive": "4b7250818647cf83",
    "add_bin": "ca6c618f1022aac7",
    "add_n_bins_random": "ac63a601ce64c3da",
    "add_n_bins_consecutive": "0f71f775f00f7d88",
    "add_route_with_removed_bins_random": "5372cce31ce2ae92",
    "add_route_with_removed_bins_consecutive": "909df4b3d10ddd73",
    "add_route_random": "dd108eb49eb33232",
    "add_route_consecutive": "a428ddd7aad5ab54",
    "rearrange_part_route": "26bd8b95af4892f7",
    "organize_route": "d592d77f0c318d21",
    "og_local_search": "19021d0c039a8690",
    "local_search_2": "7b72e913fc678832",
    "local_search_reversed": "23b60ebaba210a65",
    "insert_bins": "d7c4c48091b0a8e6",
    "remove_bins_end": "36a13fa236a24907",
    "uncross_arcs_in_sans_routes": "c8b76821ebfcb36b",
    "get_2opt_neighbors": "9c3c41c9e0b281cd",
    "relocate_within_route": "66928414af3ca584",
    "or_opt_move": "82077b01dda248d3",
    "cross_exchange": "048db83ea198da6d",
    "insert_bin_in_route": "b1369608897fb26c",
    "pt_remove_bins_from_route": "a2f1d98422894cd6",
    "pt_move_n_route_random": "352d6179c73ec164",
    "pt_move_n_route_consecutive": "518832bbfcfcf213",
    "pt_swap_n_route_random": "f896a618ed3e5797",
    "pt_swap_n_route_consecutive": "ade106315916da9b",
    "new_2opt": "5243a83835b4eb83",
    "new_move": "d76e0cfcbac41a9d",
    "new_swap": "43e69ec7161dbeb4",
    "new_remove": "e0b3db90f17ef43d",
    "new_move_n_random": "cf2508fadab615a6",
    "new_move_n_consec": "f41d0e2723669c13",
    "new_swap_n_random": "2650069003a39a56",
    "new_swap_n_consec": "77a2c1afd24a039f",
    "new_remove_n_bins": "694cc032888220b2",
    "new_remove_n_bins_consec": "0ed25eb282af14fd",
    "new_add_n_bins": "6fa437edaeada67d",
    "new_add_n_bins_consec": "5b74db92b80362c5",
    "new_add_route_removed": "fa3b580c583e7028",
    "new_add_route_removed_consec": "fa3b580c583e7028",
    "new_relocate": "531e235c8b94d292",
    "new_cross": "4466299299854dea",
    "new_or-opt": "dc5a3af24ec285d2",
    "new_insert": "05e47fcfe27464b4",
    "refine_solution": "5c2326f7365e3d4e",
    "rebalance_solution": "1dfe2b2f0a75e112",
}


def make_case(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 26))
    bins = [int(b) for b in rng.permutation(np.arange(1, n + 1))]
    n_routed = int(rng.integers(0, n + 1))
    routed, removed = bins[:n_routed], bins[n_routed:]
    k = int(rng.integers(1, 5))
    cuts = sorted(int(c) for c in rng.integers(0, n_routed + 1, size=k - 1))
    parts = [routed[a:b] for a, b in zip([0, *cuts], [*cuts, n_routed], strict=True)]
    routes = [[0, *p, 0] for p in parts]
    if rng.random() < 0.7:
        routes = [r for r in routes if len(r) > 2] or routes[:1]
    mandatory = sorted(int(b) for b in routed if rng.random() < 0.25)
    xy = rng.random((n + 1, 2)) * 10
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=-1) * rng.uniform(0.7, 1.3, (n + 1, n + 1))
    np.fill_diagonal(dist, 0.0)
    stock = [0.0] + [float(x) for x in rng.uniform(0, 60, n)]
    stocks = {i: stock[i] for i in range(1, n + 1)}
    cap = float(rng.uniform(60, 300))
    data = pd.DataFrame(
        {
            "#bin": list(range(n + 1)),
            "Stock": stock,
            "Accum_Rate": [0.0] + [float(x) for x in rng.uniform(0, 10, n)],
        }
    )
    values = {
        "E": 1.0,
        "B": 1.0,
        "R": 0.6,
        "C": 1.0,
        "V": 1.0,
        "vehicle_capacity": cap,
        "shift_duration": 400.0,
        "perc_bins_can_overflow": 0.0,
    }
    return {
        "routes": routes,
        "mandatory": mandatory,
        "removed": removed,
        "stocks": stocks,
        "cap": cap,
        "dist": dist,
        "id_to_index": {i: i for i in range(n + 1)},
        "data": data,
        "values": values,
        "n": n,
    }


def _plain(x):
    if isinstance(x, (set, frozenset)):
        return sorted(_plain(v) for v in x)
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), 9)
    return x


def _remove_bin(routes, removed, mand, rng):
    out, _ = DB.remove_n_bins_random(
        routes, removed, mand, rng, n=1, require_spare_bin=False, max_resamples=None, prune_chosen_empty_route=True
    )
    return out[0] if out else None


def _add_bin(routes, removed, rng):
    out, _ = RB.add_n_bins_random(routes, removed, rng, n=1)
    return out[0] if out else None


NEW_OPS = [
    "2opt",
    "move",
    "swap",
    "remove",
    "move_n_random",
    "move_n_consec",
    "swap_n_random",
    "swap_n_consec",
    "remove_n_bins",
    "remove_n_bins_consec",
    "add_n_bins",
    "add_n_bins_consec",
    "add_route_removed",
    "add_route_removed_consec",
    "relocate",
    "cross",
    "or-opt",
    "insert",
]


def inplace(fn, *extra, drop_ret=False):
    def run(c, rng):
        routes = c["routes"]
        ret = fn(routes, rng, *extra)
        return [routes, None if drop_ret else ret]

    return run


def pool(fn, kind):
    """LAC pool ops: routes and removed list mutated in place."""

    def run(c, rng):
        routes, removed, mand = c["routes"], c["removed"], c["mandatory"]
        if kind == "rm":
            ret = fn(routes, removed, mand, rng)
        elif kind == "add":
            ret = fn(routes, removed, rng)
        elif kind == "addroute":
            ret = fn(routes, removed, c["dist"], rng)
        else:  # split
            ret = fn(routes, c["dist"], rng)
        return [routes, removed, ret]

    return run


def new_engine(op, apply_operator):
    def run(c, rng):
        removed = set(c["removed"])
        data = {"#bin": list(range(c["n"] + 1)), "Stock": [0.0] + [c["stocks"][i] for i in range(1, c["n"] + 1)]}
        sol = apply_operator(
            op,
            copy.deepcopy(c["routes"]),
            removed,
            data,
            c["cap"],
            c["id_to_index"],
            c["stocks"],
            set(c["mandatory"]),
            c["dist"],
            rng,
        )
        return [sol, removed]

    return run


def og_local_search(fn):
    def run(c, rng):
        res = fn(c["routes"], c["removed"], c["dist"], c["mandatory"], rng, np.random.default_rng(len(c["removed"])))
        return [res, c["routes"], c["removed"]]

    return run


OPS = {
    "move_1_route": inplace(PS.move_n_route_random, 1, drop_ret=True),
    "move_2_routes": inplace(PS.move_n_2_routes_random, 1, drop_ret=True),
    "move_n_route_random": inplace(PS.move_n_route_random),
    "move_n_route_consecutive": inplace(PS.move_n_route_consecutive),
    "move_n_2_routes_random": inplace(PS.move_n_2_routes_random),
    "move_n_2_routes_consecutive": inplace(PS.move_n_2_routes_consecutive),
    "swap_1_route": inplace(PS.swap_n_route_random, 1, drop_ret=True),
    "swap_2_routes": inplace(PS.swap_n_2_routes_random, 1, drop_ret=True),
    "swap_n_route_random": inplace(PS.swap_n_route_random),
    "swap_n_route_consecutive": inplace(PS.swap_n_route_consecutive),
    "swap_n_2_routes_random": inplace(PS.swap_n_2_routes_random),
    "swap_n_2_routes_consecutive": inplace(PS.swap_n_2_routes_consecutive),
    "move_n_route_random_3": inplace(PS.move_n_route_random, 3),
    "swap_n_2_routes_consecutive_3": inplace(PS.swap_n_2_routes_consecutive, 3),
    "remove_bin": pool(_remove_bin, "rm"),
    "remove_n_bins_random": pool(DB.remove_n_bins_random, "rm"),
    "remove_n_bins_consecutive": pool(DB.remove_n_bins_consecutive, "rm"),
    "add_bin": pool(_add_bin, "add"),
    "add_n_bins_random": pool(RB.add_n_bins_random, "add"),
    "add_n_bins_consecutive": pool(RB.add_n_bins_consecutive, "add"),
    "add_route_with_removed_bins_random": pool(RB.add_route_with_removed_bins_random, "addroute"),
    "add_route_with_removed_bins_consecutive": pool(RB.add_route_with_removed_bins_consecutive, "addroute"),
    "add_route_random": pool(PS.add_route_random, "split"),
    "add_route_consecutive": pool(PS.add_route_consecutive, "split"),
    "rearrange_part_route": lambda c, rng: [PS.rearrange_part_route(c["routes"], c["dist"], rng), c["routes"]],
    "organize_route": lambda c, rng: organize_route(list(c["removed"]), c["dist"]),
    "og_local_search": og_local_search(local_search),
    "local_search_2": lambda c, rng: list(relocate_descent(c["routes"], *_fns(c), c["dist"])),
    "local_search_reversed": lambda c, rng: list(relocate_descent(c["routes"], *_fns(c), c["dist"], reverse=True)),
    "insert_bins": lambda c, rng: list(RB.insert_bins(c["routes"], c["removed"], *_fns(c))),
    "remove_bins_end": lambda c, rng: list(DB.remove_bins_end(c["routes"], c["removed"], c["mandatory"], *_fns(c))),
    "uncross_arcs_in_sans_routes": lambda c, rng: R.uncross_arcs_in_sans_routes(
        c["routes"], c["id_to_index"], c["dist"]
    ),
    "get_2opt_neighbors": lambda c, rng: [get_2opt_neighbors(r) for r in c["routes"]],
    "relocate_within_route": lambda c, rng: [PS.relocate_within_route(r, rng) for r in c["routes"]],
    "or_opt_move": lambda c, rng: [PS.or_opt_move(r, rng) for r in c["routes"]],
    "cross_exchange": lambda c, rng: PS.exchange_route_tails(c["routes"], rng),
    "insert_bin_in_route": lambda c, rng: [
        RB.insert_bin_in_route(r, b, c["id_to_index"], c["dist"]) for r in c["routes"] for b in c["removed"][:3]
    ],
    "pt_remove_bins_from_route": lambda c, rng: DB.remove_bins_from_route(c["routes"], set(c["mandatory"]), rng, 2),
    "pt_move_n_route_random": lambda c, rng: PS.transfer_n_bins_random(c["routes"], rng, 2),
    "pt_move_n_route_consecutive": lambda c, rng: PS.transfer_n_bins_consecutive(c["routes"], rng, 2),
    "pt_swap_n_route_random": lambda c, rng: PS.exchange_n_bins_random(c["routes"], rng, 2),
    "pt_swap_n_route_consecutive": lambda c, rng: (
        PS.swap_n_2_routes_consecutive(c["routes"], rng, 2, segment_may_cover_end_depot=False),
        c["routes"],
    )[1],
    **{f"new_{op}": new_engine(op, SO.apply_operator) for op in NEW_OPS},
}


def _points(c):
    rng = np.random.default_rng(c["n"])
    return [[float(x), float(y)] for x, y in rng.random((c["n"] + 1, 2))]


OPS["refine_solution"] = lambda c, rng: list(
    refine_solution(c["routes"], 1.0, 1.0, 1.0, 1.0, c["data"], _points(c), c["dist"], c["values"], iterations=2)
)
OPS["rebalance_solution"] = lambda c, rng: list(
    rebalance_solution(
        c["routes"], c["removed"], 1.0, 1.0, 1.0, 1.0, c["data"], c["mandatory"], c["dist"], c["values"], iterations=3
    )
)
DRIVERS = ["rebalance_solution", "refine_solution"]


@pytest.mark.parametrize("name", sorted(PINNED))
def test_operator_matches_pre_refactor_behaviour(name):
    records = []
    for seed in DRIVER_SEEDS if name in DRIVERS else OPERATOR_SEEDS:
        case = make_case(seed)
        rng = random.Random(seed)
        try:
            records.append({"res": _plain(OPS[name](copy.deepcopy(case), rng)), "next": rng.random()})
        except Exception as exc:  # noqa: BLE001 - edge-case errors are part of the pinned behaviour
            records.append({"error": type(exc).__name__, "next": rng.random()})
    assert hashlib.sha256(json.dumps(records).encode()).hexdigest()[:16] == PINNED[name]
