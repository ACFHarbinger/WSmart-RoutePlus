"""Tests for policy telemetry emission (§A.3)."""

import json
import time

import pytest
from logic.src.tracking.logging.modules.policy_viz_emit import (
    POLICY_VIZ_MARKER,
    STUDIO_ENV,
    PolicyVizStreamSession,
    detect_policy_viz_type,
    maybe_emit_policy_viz,
    send_policy_viz_to_gui,
)
from logic.src.tracking.viz_mixin import PolicyVizMixin

pytestmark = [pytest.mark.unit, pytest.mark.slow]


@pytest.fixture
def studio(monkeypatch):
    """Run as if WSmart-Route Studio launched the process."""
    monkeypatch.setenv(STUDIO_ENV, "1")


@pytest.fixture
def console(monkeypatch):
    """Run as a plain console or log-file run, without the Studio."""
    monkeypatch.delenv(STUDIO_ENV, raising=False)


class _DummyPolicy(PolicyVizMixin):
    def run(self) -> None:
        for i in range(3):
            self._viz_record(iteration=i, best_cost=100.0 - i)


def test_detect_policy_viz_type_alns():
    assert detect_policy_viz_type({"d_idx": [0], "best_cost": [1.0]}) == "alns"


def test_detect_policy_viz_type_hgs():
    assert detect_policy_viz_type({"generation": [0], "best_cost": [1.0]}) == "hgs"


def test_detect_policy_viz_type_generic():
    assert detect_policy_viz_type({"iteration": [0], "best_cost": [1.0]}) == "generic"


def test_maybe_emit_policy_viz_from_mixin(studio, capsys, tmp_path):
    policy = _DummyPolicy()
    policy.run()
    log_path = str(tmp_path / "sim.jsonl")

    emitted = maybe_emit_policy_viz(policy, "Test ALNS", 0, 2, log_path)
    assert emitted is True

    captured = capsys.readouterr()
    assert POLICY_VIZ_MARKER in captured.out

    content = (tmp_path / "sim.jsonl").read_text()
    assert POLICY_VIZ_MARKER in content
    payload = content.split(",", 4)[4]
    parsed = json.loads(payload)
    assert parsed["best_cost"] == [100.0, 99.0, 98.0]


def test_send_policy_viz_to_gui_skips_empty(capsys):
    send_policy_viz_to_gui({}, "P", 0, 1, None)
    assert capsys.readouterr().out == ""


def test_policy_viz_stream_session_emits_during_run(studio, capsys, tmp_path):
    policy = _DummyPolicy()
    log_path = str(tmp_path / "sim.jsonl")

    def slow_run() -> None:
        for i in range(4):
            policy._viz_record(iteration=i, best_cost=100.0 - i)
            time.sleep(0.15)

    with PolicyVizStreamSession(policy, "Stream ALNS", 0, 1, log_path):
        slow_run()

    captured = capsys.readouterr()
    assert captured.out.count(POLICY_VIZ_MARKER) >= 2
    assert (tmp_path / "sim.jsonl").exists()


def test_without_the_studio_only_the_final_snapshot_reaches_the_log(console, capsys, tmp_path):
    policy = _DummyPolicy()
    log_path = tmp_path / "sim.jsonl"

    with PolicyVizStreamSession(policy, "Console ALNS", 0, 1, str(log_path)):
        for i in range(4):
            policy._viz_record(iteration=i, best_cost=100.0 - i)
            time.sleep(0.15)

    assert POLICY_VIZ_MARKER not in capsys.readouterr().out
    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0].split(",", 4)[4])["best_cost"] == [100.0, 99.0, 98.0, 97.0]
