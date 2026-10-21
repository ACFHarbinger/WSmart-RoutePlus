"""Unit tests verifying Gemini review fixes across Lane D (Cloud State, Retry & Results)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from logic.controllers.cloud.interfaces import LaunchSpec
from logic.controllers.cloud.model import CellSpec, Resources, ResultFile, TaskRecord, TaskState
from logic.controllers.cloud.runtime import OutputMirror
from logic.controllers.cloud.service import CloudService
from logic.controllers.cloud.service_status import ServiceStatusMixin
from logic.controllers.cloud.telemetry import days_from_argv

pytestmark = [pytest.mark.unit]


def test_days_from_argv_variations():
    """Verify days_from_argv correctly parses sim.graph.n_days, sim.days, and days arguments."""
    assert days_from_argv(["test_sim", "sim.graph.n_days=10"]) == 10
    assert days_from_argv(["test_sim", "sim.days=15"]) == 15
    assert days_from_argv(["test_sim", "+sim.days=18"]) == 18
    assert days_from_argv(["test_sim", "days=20"]) == 20
    assert days_from_argv(["test_sim", "+days=25"]) == 25
    assert days_from_argv(["test_sim", "other=123"]) is None


def test_azure_batch_task_id_attempt_suffix():
    """Verify AzureBatch submit incorporates the attempt number into the task_id."""
    from logic.controllers.cloud.providers.azure.batch import AzureBatchBackend

    client = MagicMock()
    backend = AzureBatchBackend(
        client=client,
        region="eastus",
        job_id="test-job",
        job_id_spot="test-job-spot",
    )

    cell = CellSpec(cell_id="cell-1", argv=["test_sim"])
    task = TaskRecord(
        task_id="t123",
        job_id="j456",
        cell=cell,
        attempt=2,
        state=TaskState.PLACING,
    )
    launch = LaunchSpec(
        task=task,
        argv=["test_sim"],
        env={},
        secret_env={},
        resources=Resources(cpus=2, memory_gb=4, spot=False),
    )

    handle = backend.submit(launch)
    assert "t123-a2" in handle
    assert client.task.add.called
    added_task = client.task.add.call_args[1]["task"]
    assert added_task["id"] == "task-t123-a2"


def test_output_mirror_index_prunes_deleted_local_files(tmp_path: Path):
    """Verify OutputMirror.index() filters out files that were deleted locally."""
    root = tmp_path / "outputs"
    root.mkdir()
    file1 = root / "kpi.json"
    file1.write_text("{}")
    file2 = root / "checkpoint.pt"
    file2.write_text("model-weights")

    store = MagicMock()
    store.list.return_value = []
    mirror = OutputMirror(store=store, prefix="results/j1/t1", root=root)
    mirror.files["kpi.json"] = ResultFile(relpath="kpi.json", key="results/j1/t1/kpi.json", sha256="abc", size=2)
    mirror.files["checkpoint.pt"] = ResultFile(relpath="checkpoint.pt", key="results/j1/t1/checkpoint.pt", sha256="def", size=13)

    # Both exist
    index_before = mirror.index()
    assert len(index_before) == 2

    # Delete checkpoint.pt locally
    file2.unlink()
    index_after = mirror.index()
    assert len(index_after) == 1
    assert index_after[0].relpath == "kpi.json"


def test_service_requeue_transitions_directly_to_queued():
    """Verify CloudService._requeue transitions task directly to TaskState.QUEUED with attempt increment."""
    service = CloudService.__new__(CloudService)
    service.cp = MagicMock()
    service.clock = lambda: 1000.0

    task = TaskRecord(
        task_id="task-1",
        job_id="job-1",
        cell=CellSpec(cell_id="c1", argv=[]),
        attempt=1,
        state=TaskState.RUNNING,
        lease_owner="worker-1",
    )

    def fake_update(task_id, state, expected=None, **kwargs):
        return TaskRecord(
            task_id=task_id,
            job_id="job-1",
            cell=task.cell,
            attempt=kwargs.get("attempt", task.attempt),
            state=state,
            lease_owner=kwargs.get("lease_owner"),
        )

    service.cp.update_task.side_effect = fake_update
    service._event = MagicMock()

    updated = service._requeue(task, reason="preempted")
    assert updated.state == TaskState.QUEUED
    assert updated.attempt == 2
    assert updated.lease_owner is None
    assert service._event.called


def test_download_multi_cell_partitioning(tmp_path: Path):
    """Verify CloudService.download partitions multi-cell outputs into separate subdirectories."""
    service = CloudService.__new__(CloudService)
    service.store = MagicMock()
    service.cp = MagicMock()

    t1 = TaskRecord(
        task_id="t1",
        job_id="j1",
        cell=CellSpec(cell_id="cell_a", argv=[]),
        state=TaskState.SUCCEEDED,
    )
    t2 = TaskRecord(
        task_id="t2",
        job_id="j1",
        cell=CellSpec(cell_id="cell_b", argv=[]),
        state=TaskState.SUCCEEDED,
    )
    service.cp.list_tasks.return_value = [t1, t2]
    service.cp.list_results.return_value = []

    dest = tmp_path / "dest"
    dest.mkdir()

    _ = service.download("j1", dest)
    # Both tasks processed
    assert service.cp.list_results.call_count == 2


def test_latest_resource_fast_reverse_limit():
    """Verify _latest_resource requests reverse=True with limit=20 from control plane."""
    mixin = ServiceStatusMixin()
    mixin.cp = MagicMock()
    mixin.cp.get_metrics.return_value = [
        {"kind": "resource", "cpu_percent": 45.0, "memory_rss_mb": 512.0}
    ]

    res = mixin._latest_resource("task-1")
    assert res.get("cpu_percent") == 45.0
    mixin.cp.get_metrics.assert_called_with("task-1", limit=20, reverse=True)
