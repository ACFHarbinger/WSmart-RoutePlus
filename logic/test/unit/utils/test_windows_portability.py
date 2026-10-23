"""Behaviour that differed on Windows in the 1.0.2 package test (report of 2026-10-23)."""

import importlib.util
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import pytest
from logic.src.data.processor import setup as processor_setup
from logic.src.pipeline.features.eval.eval_base import samples_per_second
from logic.src.pipeline.features.test.orchestrator.parallel_runner import kill_child_processes
from logic.src.utils.functions import path as path_utils

ROOT = Path(__file__).resolve().parents[4]


def test_path_until_string_splits_on_both_separators_on_windows(monkeypatch):
    monkeypatch.setattr(path_utils, "_SEPARATORS", re.escape("\\/"))
    assert path_utils.get_path_until_string("C:\\wsr/data\\simulator\\graphs.json", "simulator") == (
        "C:\\wsr/data\\simulator"
    )
    assert path_utils.get_path_until_string("data/simulator/graphs.json", "simulator") == "data/simulator"
    assert path_utils.get_path_until_string("data/simulators/x", "simulator") is None


def test_samples_per_second_is_zero_when_the_clock_measured_nothing():
    assert samples_per_second(10, 0.0) == 0.0
    assert samples_per_second(0, 1.0) == 0.0
    assert samples_per_second(10, 2.0) == 5.0


def test_atomic_replace_waits_for_another_writer(tmp_path, monkeypatch):
    real_replace = processor_setup.os.replace
    failures = iter([PermissionError(13, "Access is denied"), PermissionError(13, "Access is denied")])

    def flaky_replace(source, destination):
        error = next(failures, None)
        if error is not None:
            raise error
        real_replace(source, destination)

    monkeypatch.setattr(processor_setup.os, "replace", flaky_replace)
    source, destination = tmp_path / "new.csv", tmp_path / "matrix.csv"
    source.write_text("new")
    processor_setup._replace_with_retry(source, destination, delay=0)
    assert destination.read_text() == "new"


def test_atomic_replace_gives_up_after_its_attempts(tmp_path, monkeypatch):
    def always_denied(source, destination):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(processor_setup.os, "replace", always_denied)
    with pytest.raises(PermissionError):
        processor_setup._replace_with_retry(tmp_path / "a", tmp_path / "b", attempts=3, delay=0)


def test_kill_child_processes_stops_live_children():
    child = mp.get_context("spawn").Process(target=time.sleep, args=(60,))
    child.start()
    try:
        kill_child_processes()
        child.join(timeout=10)
        assert not child.is_alive()
    finally:
        if child.is_alive():
            child.kill()


def test_frozen_usage_names_the_executable(monkeypatch):
    spec = importlib.util.spec_from_file_location("wsr_entry_point", ROOT / "__main__.py")
    entry_point = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry_point)
    assert entry_point._usage().startswith("Usage: python . <task>")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(Path("/opt/wsr/WSmartRoute.exe")))
    assert entry_point._usage().startswith("Usage: WSmartRoute.exe <task>")
