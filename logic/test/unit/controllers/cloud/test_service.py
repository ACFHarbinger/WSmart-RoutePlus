"""CloudService scheduling, retries, limits, cancellation and results."""

import json
import os
import signal
import sys
import textwrap
import time
from pathlib import Path

import pytest
from logic.controllers.cloud.local import SQLiteControlPlane
from logic.controllers.cloud.model import (
    CellSpec,
    ComputeKind,
    JobSpec,
    JobState,
    Provider,
    Resources,
    TaskState,
)
from logic.controllers.cloud.service import CloudService
from logic.controllers.cloud.testing import FakeClock, MemoryObjectStore, ScriptedBackend

KEY = (Provider.LOCAL, ComputeKind.PROCESS)


def _spec(n=2, **fields):
    cells = [CellSpec(cell_id=f"cell{i}", argv=["test_sim", f"sim.seed={i}"]) for i in range(n)]
    return JobSpec(name="t", cells=cells, **fields)


def _service(tmp_path, backend, clock=None):
    return CloudService(
        control_plane=SQLiteControlPlane(":memory:"),
        store=MemoryObjectStore(),
        backends={KEY: backend},
        workroot=tmp_path / "work",
        python="python",
        clock=clock or FakeClock(),
    )


def _write_outputs(service):
    def on_success(launch):
        out = Path(launch.workdir) / "outputs" / "30days" / "riomaior100_plastic" / "emp" / launch.task.cell.cell_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "log_x_1N.json").write_text(json.dumps({"task": launch.task.task_id}))

    return on_success


def _drain(service, job_id, ticks=50):
    for _ in range(ticks):
        service.tick()
        if service.status(job_id)["state"] in ("succeeded", "failed", "cancelled"):
            return service.status(job_id)
    raise AssertionError(service.status(job_id))


@pytest.mark.unit
def test_tasks_run_to_success_with_results_and_parallel_cap(tmp_path):
    backend = ScriptedBackend(polls_to_finish=2)
    service = _service(tmp_path, backend)
    backend.on_success = _write_outputs(service)
    job = service.submit(_spec(n=3, max_parallel=2))
    service.tick()
    states = [t["state"] for t in service.status(job.job_id)["tasks"]]
    assert states.count("starting") == 2 and states.count("queued") == 1
    status = _drain(service, job.job_id)
    assert status["state"] == "succeeded"
    launch = next(iter(backend.attempts.values())).launch
    assert launch.argv[:3] == ["python", "main.py", "test_sim"]
    assert "sim.resume=true" in launch.argv and "sim.checkpoint_days=1" in launch.argv
    assert launch.env["OMP_NUM_THREADS"] == "1" and launch.env["WSR_CLOUD_JOB_ID"] == job.job_id
    files = service.download(job.job_id, tmp_path / "dest")
    assert sorted(p.relative_to(tmp_path / "dest").as_posix() for p in files) == [
        f"30days/riomaior100_plastic/emp/cell{i}/log_x_1N.json" for i in range(3)
    ]
    kinds = [e.kind for e in service.cp.list_events(job.job_id)]
    assert kinds[0] == "submitted" and kinds.count("result") == 3


@pytest.mark.unit
def test_failures_retry_then_fail(tmp_path):
    service = _service(tmp_path, ScriptedBackend(script=lambda launch: "fail", polls_to_finish=0))
    job = service.submit(_spec(n=1, resources=Resources(max_retries=2)))
    status = _drain(service, job.job_id)
    assert status["state"] == "failed" and status["tasks"][0]["attempt"] == 3
    assert [e.data["attempt"] for e in service.cp.list_events(job.job_id) if e.kind == "retry"] == [2, 3]


@pytest.mark.unit
def test_preemption_resumes_on_a_new_attempt(tmp_path):
    outcomes = iter(["preempt", "succeed"])
    service = _service(tmp_path, ScriptedBackend(script=lambda launch: next(outcomes), polls_to_finish=0))
    job = service.submit(_spec(n=1))
    status = _drain(service, job.job_id)
    assert status["state"] == "succeeded" and status["tasks"][0]["attempt"] == 2
    events = [(e.data.get("from"), e.data.get("to")) for e in service.cp.list_events(job.job_id) if e.kind == "state"]
    assert ("running", "preempted") in events and ("preempted", "retrying") in events


@pytest.mark.unit
def test_no_capacity_keeps_tasks_queued(tmp_path):
    capacity = {"free": False}
    backend = ScriptedBackend(script=lambda launch: "succeed" if capacity["free"] else "no_capacity", polls_to_finish=0)
    service = _service(tmp_path, backend)
    job = service.submit(_spec(n=1))
    service.tick()
    task = service.status(job.job_id)["tasks"][0]
    assert task["state"] == "queued" and "capacity" in task["message"] and task["attempt"] == 1
    capacity["free"] = True
    assert _drain(service, job.job_id)["state"] == "succeeded"


@pytest.mark.unit
def test_wall_time_and_idle_limits(tmp_path):
    clock = FakeClock()
    service = _service(tmp_path, ScriptedBackend(script=lambda launch: "hang"), clock=clock)
    job = service.submit(_spec(n=1, resources=Resources(max_wall_seconds=100, max_retries=0)))
    service.tick()
    service.tick()
    clock.advance(101)
    service.tick()
    task = service.status(job.job_id)["tasks"][0]
    assert task["state"] == "failed" and "wall-time" in task["message"]

    job = service.submit(_spec(n=1, resources=Resources(idle_seconds=50, max_retries=0)))
    service.tick()
    clock.advance(51)
    service.tick()
    assert "no progress" in service.status(job.job_id)["tasks"][0]["message"]


@pytest.mark.unit
def test_cancel_and_manual_retry(tmp_path):
    backend = ScriptedBackend(script=lambda launch: "hang")
    service = _service(tmp_path, backend)
    job = service.submit(_spec(n=2, max_parallel=1))
    service.tick()
    service.cancel(job.job_id)
    status = _drain(service, job.job_id)
    assert status["state"] == "cancelled"
    assert sorted(t["state"] for t in status["tasks"]) == ["cancelled", "cancelled"]
    assert all(a.cancelled for a in backend.attempts.values())
    with pytest.raises(ValueError, match="clone"):
        service.retry(job.job_id)

    failing = _service(tmp_path, ScriptedBackend(script=lambda launch: "fail", polls_to_finish=0))
    job = failing.submit(_spec(n=1, resources=Resources(max_retries=0)))
    assert _drain(failing, job.job_id)["state"] == "failed"
    failing.backends[KEY].script = lambda launch: "succeed"
    assert failing.retry(job.job_id) == 1
    assert _drain(failing, job.job_id)["state"] == "succeeded"


@pytest.mark.unit
def test_submit_rejects_unknown_backends(tmp_path):
    service = _service(tmp_path, ScriptedBackend())
    with pytest.raises(ValueError, match="no backend for gcp/batch"):
        service.submit(_spec(provider=Provider.GCP, compute=ComputeKind.BATCH))


FAKE_MAIN = textwrap.dedent(
    """
    import json, os, sys, time
    args = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    out = args["sim.output_dir"]
    days = int(args["sim.graph.n_days"])
    os.makedirs(out, exist_ok=True)
    ckpt = os.path.join(out, "checkpoint_day.txt")
    start = int(open(ckpt).read()) + 1 if os.path.exists(ckpt) else 1
    with open(os.path.join(out, "starts.txt"), "a") as fh:
        fh.write(f"{start}\\n")
    for day in range(start, days + 1):
        time.sleep(float(os.environ.get("FAKE_DAY_SECONDS", "0")))
        with open(ckpt, "w") as fh:
            fh.write(str(day))
        print(f"day {day}", flush=True)
    result = os.path.join(out, f"{days}days", "riomaior100_plastic", "emp", args["sim.run_name"])
    os.makedirs(result, exist_ok=True)
    with open(os.path.join(result, "log_x_1N.json"), "w") as fh:
        json.dump({"days": days, "policy_dir": os.environ.get("WSR_POLICY_CONFIG_DIR")}, fh)
    """
)


@pytest.mark.unit
def test_cli_runs_a_manifest_end_to_end_and_resumes_a_killed_task(tmp_path, monkeypatch):
    from logic.controllers.cli import registry
    from logic.controllers.cloud import cli

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text(FAKE_MAIN)
    monkeypatch.setattr(cli, "REPO_ROOT", repo)
    monkeypatch.setenv("FAKE_DAY_SECONDS", "0.3")
    manifest = tmp_path / "cells.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "e2e",
                "resources": {"memory_gb": 0.0, "max_retries": 1},
                "cells": [
                    {
                        "cell_id": f"c{i}",
                        "command": f"WSR_POLICY_CONFIG_DIR=/yaml/{i} test_sim sim.graph.n_days=6 sim.run_name=r{i}",
                    }
                    for i in range(2)
                ],
            }
        )
    )
    root = tmp_path / "state"
    command, opts = registry.get_parser().parse_process_args(
        ["cloud", "--root", str(root), "submit", "--manifest", str(manifest), "--max-parallel", "2", "--no-worker"]
    )
    assert command == "cloud"
    job_id = cli.run_cloud(opts)["job_id"]
    service = cli.local_service(root)
    service.python = sys.executable

    # Start both tasks, then kill the first one's process group mid-run.
    service.tick()
    first = service.cp.list_tasks(job_id)[0]
    attempt = Path(first.backend_handle)
    deadline = time.time() + 20
    while not (service.task_dir(first) / "outputs" / "checkpoint_day.txt").exists():
        assert time.time() < deadline
        time.sleep(0.05)
    os.killpg(int((attempt / "pid").read_text()), signal.SIGKILL)

    assert service.run(job_id, poll_seconds=0.1, timeout=60) == JobState.SUCCEEDED
    tasks = service.cp.list_tasks(job_id)
    assert [t.attempt for t in tasks] == [2, 1]
    starts = (service.task_dir(tasks[0]) / "outputs" / "starts.txt").read_text().split()
    assert starts[0] == "1" and int(starts[1]) > 1, starts  # the retry resumed after the last finished day

    _, opts = registry.get_parser().parse_process_args(
        ["cloud", "--root", str(root), "download", job_id, "--dest", str(tmp_path / "outputs")]
    )
    cli.run_cloud(opts)
    logs = sorted((tmp_path / "outputs").rglob("log_x_1N.json"))
    assert [json.loads(p.read_text())["policy_dir"] for p in logs] == ["/yaml/0", "/yaml/1"]
    assert all(t.state == TaskState.SUCCEEDED for t in tasks)


@pytest.mark.unit
def test_status_telemetry_payload_and_policy_extraction(tmp_path):
    clock = FakeClock()
    service = _service(tmp_path, ScriptedBackend(script=lambda launch: "succeed", polls_to_finish=1), clock=clock)
    spec = JobSpec(
        name="telemetry-test",
        cells=[
            CellSpec(
                cell_id="c0",
                argv=["test_sim", "--policies", "bpc"],
                expected_wall_seconds=150.0,
            ),
            CellSpec(
                cell_id="c1",
                argv=["test_sim", "sim.policies=[alns]"],
                expected_wall_seconds=200.0,
            ),
        ],
        provider=Provider.LOCAL,
        compute=ComputeKind.PROCESS,
        resources=Resources(spot=True, machine_type="c6i.large", region="us-east-1"),
    )
    job = service.submit(spec)
    service.tick()
    clock.advance(35.0)
    service.tick()
    # Finish the running task
    clock.advance(10.0)
    service.tick()

    status = service.status(job.job_id)
    assert status["provider"] == "local"
    assert status["compute"] == "process"
    assert status["spot"] is True
    assert status["machine_type"] == "c6i.large"
    assert status["region"] == "us-east-1"
    assert len(status["tasks"]) == 2

    t0 = status["tasks"][0]
    assert t0["policy"] == "bpc"
    assert t0["expected_wall_seconds"] == 150.0
    assert t0["duration"] == 45.0
    assert t0["started_at"] is not None
    assert t0["finished_at"] is not None

    t1 = status["tasks"][1]
    assert t1["policy"] == "alns"
    assert t1["expected_wall_seconds"] == 200.0


@pytest.mark.unit
def test_status_retry_duration_and_prorated_price_estimate(tmp_path):
    from logic.controllers.cloud.service import RemoteConfig

    clock = FakeClock()
    (tmp_path / "data" / "simulator").mkdir(parents=True, exist_ok=True)
    attempts = {"c0": 0}

    def script_fn(launch):
        attempts["c0"] += 1
        if attempts["c0"] == 1:
            return "fail"
        return "succeed"

    backend = ScriptedBackend(script=script_fn, polls_to_finish=1)
    service = _service(tmp_path, backend, clock=clock)
    service.remote = RemoteConfig(
        control_plane_uri="sqlite:///control.db",
        store_uri=f"file://{tmp_path}/store",
        project_root=tmp_path,
    )
    spec = JobSpec(
        name="billing-test",
        cells=[
            CellSpec(
                cell_id="c0",
                argv=["test_sim", "--policies", "bpc"],
                expected_wall_seconds=120.0,
            ),
        ],
        provider=Provider.AWS,
        compute=ComputeKind.BATCH,
        resources=Resources(spot=True, machine_type="c6i.large", region="us-east-1", max_retries=1),
        image="release:latest",
    )
    # Register dummy backend for aws/batch
    service.backends[(Provider.AWS, ComputeKind.BATCH)] = backend
    job = service.submit(spec)

    # Attempt 1: start at 0, tick, advance 25s, tick (fails)
    service.tick()
    clock.advance(25.0)
    service.tick()

    # Attempt 2: retry, advance 80s, tick to poll and succeed
    service.tick()
    clock.advance(80.0)
    service.tick()
    service.tick()

    status = service.status(job.job_id)
    assert status["state"] == "succeeded"
    assert status["price_per_hour"] == 0.034
    assert status["on_demand_price_per_hour"] == 0.085

    task = status["tasks"][0]
    assert task["attempt"] == 2
    # Execution elapsed time is 25s + 80s = 105s
    assert task["duration"] == 105.0
    # Scheduler timing supports a prorated estimate, not provider billing minimums.
    expected_cost = round((105.0 / 3600.0) * 0.034, 6)
    assert task["cost"] == expected_cost

    assert task["started_at"] == job.created_at  # First attempt, not latest attempt.
    assert task["finished_at"] == job.created_at + 105.0
    assert status["cost_source"] == "scheduler_elapsed_price_hint"


@pytest.mark.unit
def test_retry_with_missing_history_does_not_claim_complete_duration(tmp_path):
    from logic.controllers.cloud.model import TaskRecord

    service = _service(tmp_path, ScriptedBackend())
    task = TaskRecord(
        task_id="task",
        job_id="job",
        cell=CellSpec(cell_id="c", argv=["run"]),
        state=TaskState.SUCCEEDED,
        attempt=2,
        started_at=50.0,
        finished_at=60.0,
    )
    assert service._task_duration_and_cost(task, [], 1.0, Provider.AWS) == (None, None)


@pytest.mark.unit
def test_policy_metadata_does_not_guess_from_paths_or_multi_policy_cells():
    from logic.controllers.cloud.service_status import extract_policy_from_cell

    for argv in (["test_sim", "--policies", "alns", "bpc"], ["test_sim", "sim.policies=[alns,bpc]"]):
        assert extract_policy_from_cell(CellSpec(cell_id="cell", argv=argv)) is None
    assert extract_policy_from_cell(CellSpec(cell_id="cell", argv=["test_sim"], policy_dir="/yaml/0")) is None


@pytest.mark.unit
def test_cancelled_attempt_the_backend_no_longer_knows_is_cancelled(tmp_path):
    """After a coordinator restart, a deleted (cancelled) attempt polls UNKNOWN; it must not stay running."""
    from logic.controllers.cloud.interfaces import AttemptState, BackendStatus

    class Forgetful(ScriptedBackend):
        def poll(self, handle):
            if self.attempts[handle].cancelled:
                return BackendStatus(AttemptState.UNKNOWN, message="no such attempt")
            return super().poll(handle)

    backend = Forgetful(script=lambda launch: "hang")
    service = _service(tmp_path, backend)
    job = service.submit(JobSpec(name="c", cells=[CellSpec(cell_id="x", argv=["test_sim"])]))
    service.tick()
    service.tick()
    service.cancel(job.job_id)
    for _ in range(3):
        service.tick()
    task = service.cp.list_tasks(job.job_id)[0]
    assert task.state == TaskState.CANCELLED and "no longer visible" in task.message


@pytest.mark.unit
def test_profile_service_builds_the_providers_backends(tmp_path, monkeypatch):
    from logic.controllers.cloud import cli
    from logic.controllers.cloud.auth.config import Profile, load_profiles
    from logic.controllers.cloud.providers.gcp import CloudRunJobsBackend, ComputeEngineBackend, GcpBatchBackend

    config = tmp_path / "cloud.yaml"
    config.write_text(
        "profiles:\n"
        "  gcp-test:\n"
        "    provider: gcp\n"
        "    compute: batch\n"
        "    region: europe-west4\n"
        f"    control_plane: sqlite://{tmp_path / 'cp.db'}\n"
        f"    store: file://{tmp_path / 'store'}\n"
        "    image: europe-west4-docker.pkg.dev/p/wsr/wsmart-route@sha256:1\n"
        "    backend_options:\n"
        "      project: my-project\n"
    )
    monkeypatch.setenv("WSR_CLOUD_CONFIG", str(config))
    profile = load_profiles()["gcp-test"]
    from logic.controllers.cloud.profile_service import profile_service

    service = profile_service(profile, tmp_path / "root", tmp_path)
    batch = service.backends[(Provider.GCP, ComputeKind.BATCH)]
    assert isinstance(batch, GcpBatchBackend) and (batch.project, batch.region) == ("my-project", "europe-west4")
    assert isinstance(service.backends[(Provider.GCP, ComputeKind.SERVERLESS)], CloudRunJobsBackend)
    assert isinstance(service.backends[(Provider.GCP, ComputeKind.VM)], ComputeEngineBackend)
    assert (Provider.LOCAL, ComputeKind.PROCESS) in service.backends and service.remote.store_uri.startswith("file://")
    with pytest.raises(ValueError, match="secrets"):
        Profile(
            name="bad", provider="gcp", control_plane="sqlite://x", store="file:///x", backend_options={"api_key": "x"}
        ).validate()
    with pytest.raises(ValueError, match="unknown cloud profile"):
        cli.run_cloud({"cloud_command": "status", "profile": "nope", "root": str(tmp_path / "root")})
    status = cli.run_cloud(
        {"cloud_command": "status", "profile": "gcp-test", "root": str(tmp_path / "root"), "json": True}
    )
    assert status == []
