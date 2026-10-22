"""Paper routing policies stop on an iteration budget, so their results do not depend on machine speed."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from logic.src.policies.routing.base.sub_problem import build_subproblem
from logic.src.policies.routing.meta_heuristics.hybrid_genetic_search.hgs import HGSSolver
from logic.src.policies.routing.meta_heuristics.simulated_annealing_neighborhood_search.heuristics import sans

pytestmark = [pytest.mark.unit, pytest.mark.fast]

POLICIES = Path(__file__).resolve().parents[4] / "configs" / "policies"


def _hgs(max_iterations: int) -> HGSSolver:
    rng = np.random.default_rng(3)
    pts = rng.random((9, 2))
    dist = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    wastes = {i: float(10 + i) for i in range(1, 9)}
    return HGSSolver(
        build_subproblem(dist, wastes, 40.0, 1.0, 1.0),
        {"time_limit": 600.0, "max_iterations": max_iterations, "seed": 5, "mu": 4, "lambda_param": 4},
    )


def test_hgs_stops_after_max_iterations(monkeypatch) -> None:
    solver = _hgs(7)
    calls = []
    original = solver._generate_offspring
    monkeypatch.setattr(solver, "_generate_offspring", lambda *a, **k: calls.append(1) or original(*a, **k))
    solver.solve()
    assert len(calls) == 7


def test_hgs_with_a_budget_is_repeatable() -> None:
    assert _hgs(25).solve() == _hgs(25).solve()


def test_sans_stops_after_max_iterations(monkeypatch) -> None:
    calls = []
    original = sans.apply_operator
    monkeypatch.setattr(sans, "apply_operator", lambda *a, **k: calls.append(1) or original(*a, **k))
    data = pd.DataFrame({"#bin": list(range(7)), "Stock": [50] * 7, "Accum_Rate": [1.0] * 7})
    sans.improved_simulated_annealing(
        [[1, 2, 3], [4, 5, 6]],
        np.ones((7, 7)),
        time_limit=600.0,
        id_to_index={i: i for i in range(7)},
        data=data,
        vehicle_capacity=1000.0,
        rng=random.Random(42),
        max_iterations=40,
    )
    assert 0 < len(calls) <= 40


@pytest.mark.parametrize(
    "file, path, budget_key",
    [
        ("policy_alns.yaml", ("alns", "custom"), "max_iterations"),
        ("policy_hgs.yaml", ("hgs", "custom"), "max_iterations"),
        ("policy_sans.yaml", ("sans", "new"), "max_iterations"),
        ("policy_pg_clns.yaml", ("pg_clns", "custom"), "max_iterations"),
        ("policy_aco_hh.yaml", ("aco_hh", "custom"), "max_iterations"),
        ("policy_psoma.yaml", ("psoma",), "max_iterations"),
        ("other/ri_cls.yaml", ("default",), "iterations"),
    ],
)
def test_paper_policies_keep_time_limit_as_a_guard(file, path, budget_key) -> None:
    node = yaml.safe_load((POLICIES / file).read_text())
    for key in path:
        node = node[key]
    assert int(node[budget_key]) > 0
    assert float(node["time_limit"]) >= 600.0
