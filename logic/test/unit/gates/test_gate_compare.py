"""Unit tests for the Q-14 gate comparators and override resolver (WP-12, #117).

Pure JSON/dict inputs only: no simulator, no Gurobi, no torch. Fast.
"""

import math
import os
import sys
from pathlib import Path

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
    # The packing file is pk_* after the WP-03 rename and ms_* on the unchanged baseline.
    renamed = (Path(__file__).resolve().parents[4] / "logic/configs/policies/other/pk_last_minute.yaml").exists()
    packing = {f"other/{'pk' if renamed else 'ms'}_last_minute.yaml": ["last_minute_cf70"]}
    entry = cell_policy_entry("ALNS")
    assert entry == {
        "alns": {
            "packing": packing,
            "route_improvement": {"other/ri_cls.yaml": ["default"]},
            "time_limit": 0,
        }
    }
    hgs = cell_policy_entry("HGS")
    assert hgs["hgs"]["restart_timer"] == 0
    assert cell_policy_entry("CVRP") == {
        "cvrp": {
            "packing": packing,
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


def test_certificates_require_equal_finite_bounds():
    assert not compare_certificates(100, 100, 100, 101).passed
    assert not compare_certificates(100, math.inf, 100, math.inf).passed
    assert not compare_tolerance(1, 0, 1, 0, math.inf, 0).passed


def test_physical_time_is_checked_while_wall_time_is_excluded():
    assert compare_exact({"time_spent": 1}, {"time_spent": 2})
    assert not compare_exact({"day_wall_time": 1}, {"day_wall_time": 2})


def test_canonical_log_kpi_key_and_time_capped_overrides():
    assert extract_cell_kpis({"kg/km": 4, "overflows": 2}) == (4, 2)
    entry = cell_policy_entry("CVRP", time_capped=True)["cvrp"]
    assert entry["time_limit"] == 60 and entry["solution_limit"] is None
    assert cell_policy_entry("ALNS", time_capped=True)["alns"]["time_limit"] == 60


def test_calibration_rejects_insufficient_or_nonfinite_evidence():
    with pytest.raises(ValueError):
        freeze_tolerances([])
    with pytest.raises(ValueError):
        freeze_tolerances([{"kgkm": math.nan, "overflows": 0}] * 2)


def test_gate_a_cannot_pass_empty_sweeps(tmp_path):
    from argparse import Namespace

    from tools.gates.run_gate import cmd_gate_a

    with pytest.raises(SystemExit, match="FAILED"):
        cmd_gate_a(Namespace(baseline=str(tmp_path), candidate=str(tmp_path)))


def test_missing_bound_is_not_fabricated():
    from tools.gates.run_gate import _bound

    with pytest.raises(KeyError):
        _bound({"samples": {"s0": {"profit": 10}}})


def test_log_lookup_finds_nested_lowercase_policy_and_rejects_duplicates(tmp_path):
    import json

    from tools.gates.run_gate import _load_policy_log

    nested = tmp_path / "5days" / "riomaior100" / "emp"
    nested.mkdir(parents=True)
    path = nested / "log_last_minute_cf70_alns_custom_cls_1N.json"
    path.write_text(json.dumps(_log([0, 1, 0])))
    assert _load_policy_log(str(tmp_path), "ALNS")["samples"]
    (nested / "log_other_alns_custom_1N.json").write_text(path.read_text())
    with pytest.raises(ValueError, match="found 2"):
        _load_policy_log(str(tmp_path), "ALNS")


def test_heavy_lock_is_retained(monkeypatch, tmp_path):
    import fcntl

    from tools.gates import run_gate

    monkeypatch.setattr(run_gate, "HEAVY_LOCK", str(tmp_path / "heavy.lock"))
    monkeypatch.setattr(run_gate, "_HEAVY_LOCK_HANDLE", None)
    monkeypatch.setattr(run_gate, "chain_done", lambda: True)
    run_gate.require_heavy()  # no heavy execution; only exercise lock lifetime
    try:
        with open(run_gate.HEAVY_LOCK, "a") as other, pytest.raises(BlockingIOError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        run_gate._HEAVY_LOCK_HANDLE.close()


def test_gate_b_rejects_empty_candidates(tmp_path):
    import json
    from argparse import Namespace

    from tools.gates.cells import BASELINE_COMMIT
    from tools.gates.run_gate import cmd_gate_b

    archive = tmp_path / "archive.csv"
    candidate = tmp_path / "candidate.csv"
    frozen = tmp_path / "frozen.json"
    archive.write_text("constructor,city,N,kgkm,overflows\nALNS,Rio Maior,100,4,0\n")
    candidate.write_text("constructor,city,N,kgkm,overflows\n")
    frozen.write_text(json.dumps({"baseline_commit": BASELINE_COMMIT, "cells": {}}))
    with pytest.raises(SystemExit, match="incomplete sweep"):
        cmd_gate_b(Namespace(archived=str(archive), candidate=str(candidate), tolerances=str(frozen)))


def test_failed_samples_cannot_pass_as_identical_artifacts(tmp_path):
    import json

    from tools.gates.run_gate import _load_policy_log

    path = tmp_path / "log_alns_1N.json"
    log = _log([0])
    log["samples"]["s0"]["status"] = "failed"
    path.write_text(json.dumps(log))
    with pytest.raises(ValueError, match="Failed samples"):
        _load_policy_log(str(tmp_path), "ALNS")


def test_chain_gate_reads_shared_checkout_log_from_a_worktree(monkeypatch, tmp_path):
    import subprocess

    from tools.gates import run_gate

    common = tmp_path / "shared" / ".git"
    common.mkdir(parents=True)
    log = common.parent / ".agent/cache/train-logs/chain.log"
    log.parent.mkdir(parents=True)
    log.write_text("sim chain done\n")
    monkeypatch.setattr(run_gate, "CHAIN_LOG", ".agent/cache/train-logs/chain.log")
    monkeypatch.setattr(run_gate.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, str(common)))
    assert run_gate.chain_done()


def test_sans_cell_has_a_real_work_budget():
    # SANS stops at once on time_limit 0 (no > 0 guard); its schedule bounds the work instead.
    entry = cell_policy_entry("SANS")["sans"]
    assert entry["time_limit"] > 1e6 and entry["iterations_per_T"] == 200
    capped = cell_policy_entry("SANS", time_capped=True)["sans"]
    assert capped["time_limit"] == 60.0 and "iterations_per_T" not in capped
