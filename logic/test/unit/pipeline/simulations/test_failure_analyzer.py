"""Tests for FailureAnalyzer and emit bridge (§A.6)."""

import json

import numpy as np
import pandas as pd
import pytest
from logic.src.constants import MAX_CAPACITY_PERCENT
from logic.src.tracking.logging.modules.failure_analyzer import FailureAnalyzer
from logic.src.tracking.logging.modules.failure_emit import (
    SIM_FAILURE_MARKER,
    emit_sim_failure_summary,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def _coords(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"ID": [0, 101, 102, 103][: n + 1]})


def test_no_failure_on_clean_day():
    summary = FailureAnalyzer().analyze(
        new_overflows=0,
        sum_lost=0.0,
        profit=10.0,
        fill=np.array([4.0, 5.0, 3.0]),
        total_fill=np.array([40.0, 50.0, 30.0]),
        bins_means=np.array([4.0, 5.0, 3.0]),
        bins_real_c=np.array([40.0, 50.0, 30.0]),
        tour=[0, 1, 2, 0],
        collected=np.array([1.0, 1.0, 0.0]),
        coords=_coords(),
    )
    assert summary["has_failure"] is False
    assert summary["root_causes"] == []
    assert summary["overflow_bins"] == []


def test_overflow_and_fill_spike_detected():
    summary = FailureAnalyzer().analyze(
        new_overflows=1,
        sum_lost=12.5,
        profit=-3.0,
        fill=np.array([20.0, 4.0]),
        total_fill=np.array([100.0, 40.0]),
        bins_means=np.array([4.0, 4.0]),
        bins_real_c=np.array([MAX_CAPACITY_PERCENT, 40.0]),
        tour=[0],
        collected=np.zeros(2),
        coords=_coords(2),
    )
    assert summary["has_failure"] is True
    assert "overflow_event" in summary["root_causes"]
    assert "waste_lost" in summary["root_causes"]
    assert "negative_profit" in summary["root_causes"]
    assert "fill_rate_spike" in summary["root_causes"]
    assert len(summary["overflow_bins"]) == 1
    assert summary["overflow_bins"][0]["bin_id"] == 101
    assert summary["overflow_bins"][0]["fill_spike"] is True


def test_skipped_high_fill_bins():
    summary = FailureAnalyzer().analyze(
        new_overflows=0,
        sum_lost=0.0,
        profit=-1.0,
        fill=np.array([2.0, 2.0]),
        total_fill=np.array([30.0, 95.0]),
        bins_means=np.array([2.0, 2.0]),
        bins_real_c=np.array([30.0, 95.0]),
        tour=[0, 1, 0],
        collected=np.array([0.5, 0.0]),
        coords=_coords(2),
    )
    assert "negative_profit" in summary["root_causes"]
    assert "skipped_high_fill" in summary["root_causes"]
    assert len(summary["skipped_high_fill_bins"]) == 1
    assert summary["skipped_high_fill_bins"][0]["bin_id"] == 102


def test_emit_sim_failure_summary_stdout(capsys):
    emit_sim_failure_summary(
        {"has_failure": True, "severity": "warning", "root_causes": ["overflow_event"], "summary": "test"},
        "greedy",
        0,
        2,
    )
    captured = capsys.readouterr()
    assert SIM_FAILURE_MARKER in captured.out
    payload = json.loads(captured.out.strip().split(SIM_FAILURE_MARKER, 1)[1].split(",", 3)[3])
    assert payload["severity"] == "warning"


def test_emit_skips_clean_day(capsys):
    emit_sim_failure_summary({"has_failure": False}, "greedy", 0, 1)
    captured = capsys.readouterr()
    assert SIM_FAILURE_MARKER not in captured.out


def test_emit_sim_failure_summary_jsonl(tmp_path):
    log_path = str(tmp_path / "sim.jsonl")
    emit_sim_failure_summary(
        {"has_failure": True, "severity": "critical", "root_causes": ["overflow_event"], "summary": "overflow"},
        "ALNS",
        1,
        3,
        log_path=log_path,
    )
    content = (tmp_path / "sim.jsonl").read_text()
    assert SIM_FAILURE_MARKER in content
    assert "ALNS" in content
