"""Unit tests for the Q-14 gate comparators and override resolver (WP-12, #117).

Pure JSON/dict inputs only: no simulator, no Gurobi, no torch. Fast.
"""

import math
import os
import sys

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from tools.gates.cells import GATE_A_CELLS, cell_policy_entry, get_cell  # noqa: E402
from tools.gates.compare import (  # noqa: E402
    CERT_TOL,
    check_repeats_identical,
    compare_certificates,
    compare_exact,
    compare_tolerance,
    extract_cell_kpis,
    freeze_tolerances,
    gate_b_verdict,
    strip_excluded,
)


def _log(tour, kgkm=8.5, overflows=2.0, time=12.0):
    return {
        "samples": {"s0": {"kgkm": kgkm, "overflows": overflows, "time": time}},
        "daily": [
            {"day": 0, "tour": tour, "kg": 100.0, "time": 3.0},
            {"day": 1, "tour": tour[::-1], "kg": 120.0, "time": 4.0},
        ],
    }


def test_strip_excluded_removes_timing_recursively():
    cleaned = strip_excluded(_log([0, 1, 0]))
    assert cleaned["samples"]["s0"] == {"kgkm": 8.5, "overflows": 2.0}
    assert cleaned["daily"][0] == {"day": 0, "tour": [0, 1, 0], "kg": 100.0}


def test_compare_exact_passes_on_identical_logs():
    assert compare_exact(_log([0, 1, 0]), _log([0, 1, 0])) == []


def test_compare_exact_ignores_timing_only_differences():
    assert compare_exact(_log([0, 1, 0], time=1.0), _log([0, 1, 0], time=999.0)) == []


def test_compare_exact_reports_tour_diff_with_path():
    diffs = compare_exact(_log([0, 1, 0]), _log([0, 2, 0]))
    assert any("tour" in d.path for d in diffs)
    assert diffs[0].expected != diffs[0].actual


def test_compare_exact_reports_missing_and_extra_keys():
    diffs = compare_exact({"a": 1}, {"b": 1})
    assert {d.path for d in diffs} == {"$.a", "$.b"}


def test_compare_exact_treats_nan_as_equal():
    assert compare_exact({"v": math.nan}, {"v": math.nan}) == []
    assert compare_exact({"v": math.nan}, {"v": 1.0}) != []


def test_extract_cell_kpis():
    assert extract_cell_kpis({"kgkm": 8.25, "overflows": 3.0}) == (8.25, 3.0)


def test_compare_tolerance_within_and_outside():
    assert compare_tolerance(8.0, 2.0, 8.05, 2.0, 0.1, 0.0).passed
    assert not compare_tolerance(8.0, 2.0, 8.5, 2.0, 0.1, 0.0).passed
    # Better results outside tolerance also need review (either direction).
    assert not compare_tolerance(8.0, 2.0, 7.5, 2.0, 0.1, 0.0).passed
    assert not compare_tolerance(8.0, 2.0, 8.0, 1.0, 0.1, 0.0).passed


def test_compare_certificates():
    assert compare_certificates(100.0, 100.0, 100.0 + 1e-7, 100.0).passed
    # Incumbent outside 1e-6 fails.
    assert not compare_certificates(100.0, 100.0, 100.01, 100.01).passed
    # Invalid certificate (bound below incumbent) fails.
    assert not compare_certificates(100.0, 99.0, 100.0, 100.0).passed
    assert CERT_TOL == 1e-6


def test_freeze_tolerances_max_pairwise():
    frozen = freeze_tolerances(
        [
            {"kgkm": 8.0, "overflows": 2.0},
            {"kgkm": 8.1, "overflows": 2.0},
            {"kgkm": 8.05, "overflows": 3.0},
        ]
    )
    assert frozen == {"tol_kgkm": pytest.approx(0.1), "tol_overflows": 1.0}


def test_check_repeats_identical():
    assert check_repeats_identical([_log([0, 1, 0])] * 3) == []
    diffs = check_repeats_identical([_log([0, 1, 0]), _log([0, 2, 0])])
    assert diffs and diffs[0].path.startswith("repeat2:")


def test_gate_b_verdict_pass_review_and_missing_key():
    row = {"kgkm": "8.0", "overflows": "2.0"}
    tol = {"tol_kgkm": 0.1, "tol_overflows": 0.0}
    assert gate_b_verdict(row, dict(row), tol, "ALNS").passed
    assert not gate_b_verdict(row, {"kgkm": "9.0", "overflows": "2.0"}, tol).passed
    assert not gate_b_verdict(row, {"kgkm": "8.0"}, tol).passed


def test_cell_policy_entry_pins_packing_improver_and_timing():
    entry = cell_policy_entry("ALNS")
    assert entry == {
        "alns": {
            "packing": {"other/ms_last_minute.yaml": ["last_minute_cf70"]},
            "route_improvement": {"other/ri_cls.yaml": ["default"]},
            "time_limit": 0,
        }
    }
    hgs = cell_policy_entry("HGS")
    assert hgs["hgs"]["restart_timer"] == 0
    assert cell_policy_entry("CVRP") == {
        "cvrp": {
            "packing": {"other/ms_last_minute.yaml": ["last_minute_cf70"]},
            "route_improvement": {"other/ri_cls.yaml": ["default"]},
            "solution_limit": 1000,
        }
    }


def test_cell_policy_entry_am_needs_checkpoint():
    with pytest.raises(ValueError, match="model-path"):
        cell_policy_entry("AM")
    entry = cell_policy_entry("AM", model_path="checkpoints/am0")
    assert entry["na"]["model_path"] == "checkpoints/am0"


def test_gate_cells_cover_released_routing():
    names = {c["constructor"] for c in GATE_A_CELLS}
    assert names == {
        "SWC-TCF",
        "BPC",
        "ALNS",
        "SANS",
        "HGS",
        "PSOMA",
        "ACO-HH",
        "PG-CLNS",
        "AM",
        "CVRP",
    }
    assert get_cell("bpc")["compare_mode"] == "certificate_fallback"
    assert get_cell("cvrp")["overrides"] == {"solution_limit": 1000}
    assert get_cell("hgs")["overrides"] == {"time_limit": 0, "restart_timer": 0}
