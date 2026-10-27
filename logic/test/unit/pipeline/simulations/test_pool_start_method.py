"""The simulator pool's start method on platforms with and without fork."""

from types import SimpleNamespace

import pytest
from logic.src.pipeline.features.test.orchestrator import parallel_runner

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.mark.parametrize(
    "methods, no_cuda, device, expected",
    [
        (["fork", "spawn", "forkserver"], True, "cpu", "fork"),
        (["fork", "spawn", "forkserver"], False, "cpu", "spawn"),
        (["fork", "spawn", "forkserver"], True, "cuda:0", "spawn"),
        (["spawn"], True, "cpu", "spawn"),  # Windows: no fork
    ],
)
def test_start_method_falls_back_to_spawn_without_fork(monkeypatch, methods, no_cuda, device, expected):
    monkeypatch.setattr(parallel_runner.mp, "get_all_start_methods", lambda: methods)
    assert parallel_runner.pool_start_method(SimpleNamespace(no_cuda=no_cuda), device) == expected


def test_package_entry_point_calls_freeze_support():
    from pathlib import Path

    entry = Path(__file__).resolve().parents[5] / "__main__.py"
    assert "multiprocessing.freeze_support()" in entry.read_text()
