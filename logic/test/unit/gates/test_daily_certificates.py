"""Certificate fallback requires evidence for every paired sample and day."""

import copy

import pytest
from logic.gates.compare import compare_daily_certificates
from logic.src.pipeline.simulations.reporting import (
    current_solver_certificate,
    note_root_bound,
    note_solver_incumbent,
    reset_solver_status,
)


def _log():
    return {
        "samples": {"0": {"profit": 100.0}},
        "daily": {
            "0": {
                "day": [1, 2],
                "solver_certificate": [
                    {"incumbent": 20.0, "bound": 21.0, "instance": "instance-day-1"},
                    {"incumbent": 80.0, "bound": 80.0, "instance": "instance-day-2"},
                ],
            }
        },
    }


def test_every_day_is_compared_instead_of_horizon_totals():
    a = _log()
    assert compare_daily_certificates(a, copy.deepcopy(a)).passed
    b = copy.deepcopy(a)
    b["daily"]["0"]["solver_certificate"][1]["bound"] += 1.0
    assert not compare_daily_certificates(a, b).passed


@pytest.mark.parametrize("change", ["missing", "short", "none", "nan", "days", "sample"])
def test_unpaired_or_incomplete_evidence_fails(change):
    a, b = _log(), _log()
    daily = b["daily"]["0"]
    if change == "missing":
        daily.pop("solver_certificate")
    if change == "short":
        daily["solver_certificate"].pop()
    if change == "none":
        daily["solver_certificate"][0] = None
    if change == "nan":
        daily["solver_certificate"][0]["bound"] = float("nan")
    if change == "days":
        daily["day"] = [1, 1]
    if change == "sample":
        b["samples"]["1"] = {}
    assert not compare_daily_certificates(a, b).passed


def test_proof_is_reset_between_constructed_days():
    reset_solver_status()
    assert current_solver_certificate() is None
    note_root_bound(12.0)
    assert current_solver_certificate() is None
    note_solver_incumbent(10.0)
    assert current_solver_certificate()["bound"] == 12.0
    reset_solver_status()
    note_solver_incumbent(10.0)
    assert current_solver_certificate() is None


def test_equal_objectives_on_different_construction_inputs_fail():
    a, b = _log(), _log()
    b["daily"]["0"]["solver_certificate"][0]["instance"] = "different-input"
    assert not compare_daily_certificates(a, b).passed


def test_fingerprint_changes_with_constraints_and_objective():
    import numpy as np
    from logic.src.utils.routing.problem_fingerprint import problem_fingerprint

    d = np.array([[0.0, 1.0], [1.0, 0.0]])
    baseline = problem_fingerprint(d, {1: 3.0}, 4.0, 1.0, 2.0, [1], None)
    assert baseline == problem_fingerprint(d.copy(), {1: 3.0}, 4.0, 1.0, 2.0, {1}, 0)
    assert baseline != problem_fingerprint(d, {1: 4.0}, 4.0, 1.0, 2.0, [1], None)
    assert baseline != problem_fingerprint(d, {1: 3.0}, 4.0, 1.0, 2.0, [], None)
    assert baseline != problem_fingerprint(d, {1: 3.0}, 4.0, 1.0, 2.0, [1], 1)
    assert baseline != problem_fingerprint(d, {1: 3.0}, 4.0, 2.0, 2.0, [1], None)
