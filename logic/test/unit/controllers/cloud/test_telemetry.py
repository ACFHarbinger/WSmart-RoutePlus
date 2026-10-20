"""Task instrumentation (CR-12): resource and progress records, batching, the tracking import and the CLI."""

import json
import textwrap
from pathlib import Path

import pytest
from logic.controllers.cloud.local import FilesystemObjectStore, SQLiteControlPlane
from logic.controllers.cloud.model import CellSpec, ComputeKind, Event, JobRecord, JobSpec, Provider, TaskRecord
from logic.controllers.cloud.runtime import run_task
from logic.controllers.cloud.service import CloudService
from logic.controllers.cloud.telemetry import (
    ProgressReader,
    TaskTelemetry,
    TelemetryBuffer,
    days_from_argv,
    import_job,
    task_metrics,
)
from logic.controllers.cloud.testing import FakeClock, MemoryObjectStore
from logic.src.tracking.core.store import TrackingStore

# A stand-in simulator that writes the day logger's realtime lines (gui.py format), one per day.
FAKE_MAIN = textwrap.dedent(
    """
    import json, os, sys, time
    args = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    out = args["sim.output_dir"]
    days = int(args["sim.graph.n_days"])
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "log_realtime_emp_1N.jsonl")
    for day in range(1, days + 1):
        x = [0.0] * 2_000_000  # some resident memory to sample
        with open(path, "a") as fh:
            payload = {"kg": 10.0 * day, "km": 2.0, "kg/km": 5.0 * day, "overflows": 0, "solver_status": "ok"}
            fh.write("GUI_DAY_LOG_START:LA ALNS,0,%d,%s\\n" % (day, json.dumps(payload)))
        time.sleep(float(os.environ.get("FAKE_DAY_SECONDS", "0.3")))
    if os.environ.get("FAKE_FAIL"):
        print("boom: the solver crashed")
        sys.exit(2)
    """
)


def _line(policy, day, **kpis):
    return f"GUI_DAY_LOG_START:{policy},0,{day},{json.dumps(kpis)}\n"


def _job(tmp_path, days=3):
    root = tmp_path / "app"
    root.mkdir(parents=True)
    (root / "main.py").write_text(FAKE_MAIN)
    cell = CellSpec(cell_id="c/0", argv=["test_sim", f"sim.graph.n_days={days}"])
    spec = JobSpec(name="tel", cells=[cell], provider=Provider.GCP, compute=ComputeKind.BATCH, image="img")
    cp = SQLiteControlPlane(str(tmp_path / "cp.db"))
    cp.create_job(JobRecord(job_id="job-1", spec=spec), [TaskRecord(task_id="task-1", job_id="job-1", cell=cell)])
    return cp, FilesystemObjectStore(tmp_path / "store"), root, tmp_path / "work"


@pytest.mark.unit
def test_progress_reader_reads_new_complete_lines_only(tmp_path):
    reader = ProgressReader(tmp_path, days=30, clock=lambda: 7.0)
    log = tmp_path / "sub" / "log_realtime_emp_1N.jsonl"
    log.parent.mkdir()
    log.write_text(
        _line("LA ALNS", 1, kg=5.0, km=1.0, solver_status="gurobi:OPTIMAL") + 'GUI_DAY_LOG_START:LA ALNS,0,2,{"kg"'
    )
    first = reader.read()
    assert [(r["policy"], r["day"], r["days"], r["kpis"]) for r in first] == [
        ("LA ALNS", 1, 30, {"kg": 5.0, "km": 1.0})
    ]
    assert first[0]["solver_status"] == "gurobi:OPTIMAL" and reader.latest == {
        "day": 1,
        "days": 30,
        "policy": "LA ALNS",
    }
    with open(log, "a") as fh:  # the half-written line is completed, then another day follows
        fh.write(": 6.0}\n" + "not a day line\n" + _line("LA ALNS", 3, kg=7.0))
    assert [r["day"] for r in reader.read()] == [2, 3]
    assert reader.read() == []


@pytest.mark.unit
def test_buffer_batches_and_rate_limits_writes():
    cp = SQLiteControlPlane(":memory:")
    now = {"t": 100.0}
    buffer = TelemetryBuffer(cp, "t1", flush_seconds=10.0, max_batch=3, clock=lambda: now["t"])
    buffer.add([{"at": i, "kind": "resource"} for i in range(5)])
    assert buffer.flush() == 3 and len(cp.get_metrics("t1")) == 3
    now["t"] += 5
    assert buffer.flush() == 0  # within the interval
    now["t"] += 6
    buffer.add([{"at": 9, "kind": "resource"}, {}])  # empty records are dropped
    assert buffer.flush() == 3 and buffer.flush(force=True) == 0
    assert buffer.written == 6 and len(cp.get_metrics("t1")) == 6


@pytest.mark.unit
def test_days_come_from_the_command_line():
    assert days_from_argv(["test_sim", "sim.graph.n_days=90", "x=1"]) == 90
    assert days_from_argv(["test_sim", "+sim.graph.n_days=7"]) == 7
    assert days_from_argv(["test_sim"]) is None


@pytest.mark.unit
def test_telemetry_without_psutil_still_reports_progress(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    cp = SQLiteControlPlane(":memory:")
    (tmp_path / "log_realtime_emp_1N.jsonl").write_text(_line("P", 1, kg=1.0))
    telemetry = TaskTelemetry(cp, "t1", tmp_path, tmp_path, days=2, clock=lambda: 50.0)
    telemetry.attach(12345)
    assert telemetry.sampler is None
    telemetry.tick()
    telemetry.close()
    assert [r["kind"] for r in cp.get_metrics("t1")] == ["progress"] and telemetry.errors == []


@pytest.mark.unit
def test_runtime_streams_resource_and_progress_records(tmp_path):
    cp, store, root, work = _job(tmp_path, days=3)
    assert run_task(cp, store, "job-1", "task-1", work, project_root=root, sync_seconds=0.05, sample_seconds=0.1) == 0
    records = cp.get_metrics("task-1")
    progress = [r for r in records if r["kind"] == "progress"]
    resource = [r for r in records if r["kind"] == "resource"]
    assert [r["day"] for r in progress] == [1, 2, 3] and progress[-1]["kpis"]["kg"] == 30.0
    assert resource and all(r["memory_rss_mb"] > 0 and r["processes"] >= 1 for r in resource)
    task = cp.get_task("task-1")
    assert task.progress["done"] is True and task.progress["day"] == 3 and task.progress["days"] == 3

    payload = task_metrics(records, "task-1")
    assert payload["progress"] == {"day": 3, "days": 3, "policy": "LA ALNS", "kpis": progress[-1]["kpis"]}
    assert (
        len(payload["points"]) == len(resource) and payload["peak_memory_mb"] >= payload["points"][0]["memory_rss_mb"]
    )
    assert [d["day"] for d in payload["days"]] == [1, 2, 3] and payload["days"][0]["solver_status"] == "ok"


@pytest.mark.unit
def test_failed_task_records_its_log_tail(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    monkeypatch.setenv("FAKE_DAY_SECONDS", "0")
    cp, store, root, work = _job(tmp_path, days=1)
    assert run_task(cp, store, "job-1", "task-1", work, project_root=root, sync_seconds=0.05) == 2
    tails = [e for e in cp.list_events("job-1") if e.kind == "log_tail"]
    assert len(tails) == 1 and tails[0].data["code"] == 2 and "boom" in tails[0].data["tail"]


def _service(tmp_path):
    clock = FakeClock()
    return CloudService(
        control_plane=SQLiteControlPlane(":memory:"),
        store=MemoryObjectStore(),
        backends={},
        workroot=tmp_path / "work",
        clock=clock,
    )


def _imported_job(service):
    cell = CellSpec(cell_id="cell/a", argv=["test_sim", "sim.graph.n_days=2"])
    spec = JobSpec(name="imp", cells=[cell], provider=Provider.LOCAL, compute=ComputeKind.PROCESS)
    cp = service.cp
    cp.create_job(JobRecord(job_id="job-9", spec=spec), [TaskRecord(task_id="task-9", job_id="job-9", cell=cell)])
    cp.put_metrics(
        "task-9",
        [
            {"at": 1.0, "kind": "resource", "cpu_percent": 90.0, "memory_rss_mb": 300.0, "threads": 4},
            {"at": 2.0, "kind": "progress", "policy": "P", "sample": 0, "day": 1, "kpis": {"kg": 5.0}},
            {"at": 3.0, "kind": "resource", "cpu_percent": 50.0, "memory_rss_mb": 320.0, "threads": 4},
            {"at": 4.0, "kind": "cost", "accrued_usd": 0.25},
        ],
    )
    cp.append_event(Event(job_id="job-9", kind="state", task_id="task-9", data={"to": "running"}, at=1.0))
    cp.append_event(Event(job_id="job-9", kind="submitted", data={}, at=0.5))
    return cp


@pytest.mark.unit
def test_import_job_writes_tracking_runs_and_is_repeatable(tmp_path):
    service = _service(tmp_path)
    cp = _imported_job(service)
    store = TrackingStore(str(tmp_path / "test_tracking" / "tracking.db"))
    for _ in range(2):
        run_ids = import_job(cp, "job-9", store)
    assert run_ids == ["cloud-task-9"]
    tags = store.get_tags("cloud-task-9")
    assert tags["job_id"] == "job-9" and tags["cell_id"] == "cell/a" and tags["provider"] == "local"
    history = store.get_metric_history("cloud-task-9", "resource/memory_rss_mb")
    assert [h["value"] for h in history] == [300.0, 320.0]  # not doubled by the second import
    assert [h["value"] for h in store.get_metric_history("cloud-task-9", "progress/kg")] == [5.0]
    assert [h["value"] for h in store.get_metric_history("cloud-task-9", "cost/accrued_usd")] == [0.25]
    assert [e["kind"] for e in store.get_run_events("cloud-task-9")] == ["submitted", "state"]


@pytest.mark.unit
def test_service_and_cli_expose_task_metrics(tmp_path):
    service = _service(tmp_path)
    _imported_job(service)
    status = service.status("job-9")["tasks"][0]
    assert status["cpu_percent"] == 50.0 and status["memory_rss_mb"] == 320.0
    metrics = service.metrics("task-9")
    assert metrics["peak_cpu"] == 90.0 and metrics["peak_memory_mb"] == 320.0
    assert [p["timestamp_ms"] for p in metrics["points"]] == [1000, 3000]
    assert service.metrics("task-9", since=2.5)["points"] == [
        {"timestamp_ms": 3000, "cpu_percent": 50.0, "memory_rss_mb": 320.0}
    ]

    from logic.controllers.cloud import cli

    ctx = cli._Context({"task_id": "task-9", "json": True}, tmp_path, tmp_path, service)
    assert cli._COMMANDS["metrics"](ctx)["progress"]["day"] == 1
    ctx = cli._Context(
        {"job_id": "job-9", "json": True, "tracking_uri": str(tmp_path / "test_tracking")}, tmp_path, tmp_path, service
    )
    assert cli._COMMANDS["track"](ctx) == ["cloud-task-9"]
    # a path with "test_tracking" is exempt from conftest's tracking.db redirect, so each test has its own database
    run = TrackingStore(str(tmp_path / "test_tracking" / "tracking.db")).get_run("cloud-task-9")
    assert run is not None and Path(run["artifact_dir"]) == (tmp_path / "results" / "job-9").resolve()
