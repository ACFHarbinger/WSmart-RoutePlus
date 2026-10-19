"""Task runtime (CR-02): input staging, output mirroring, preemption, idle kill and Gurobi licenses."""

import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from logic.controllers.cloud.local import FilesystemObjectStore, SQLiteControlPlane
from logic.controllers.cloud.model import (
    CellSpec,
    ComputeKind,
    GurobiMode,
    JobRecord,
    JobSpec,
    Provider,
    Resources,
    TaskRecord,
)
from logic.controllers.cloud.profiles import control_plane_from_uri, store_from_uri
from logic.controllers.cloud.runtime import EXIT_IDLE, EXIT_NO_LICENSE, EXIT_TERMINATED, run_task
from logic.controllers.cloud.service import CloudService, RemoteConfig
from logic.controllers.cloud.staging import fetch_inputs, stage_job_inputs
from logic.controllers.cloud.testing import FakeClock, MemoryObjectStore, ScriptedBackend

REPO = Path(__file__).resolve().parents[5]

# A stand-in simulator: one checkpoint per day, resumes from it, and writes a result that
# depends only on the days simulated, so interrupted and uninterrupted runs must match.
FAKE_MAIN = textwrap.dedent(
    """
    import json, os, sys, time
    args = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    out = args["sim.output_dir"]
    days = int(args["sim.graph.n_days"])
    data = open(os.path.join("data", "simulator", "datasets", "d.npz")).read()
    policy = open(os.path.join(os.environ["WSR_POLICY_CONFIG_DIR"], "policy_x.yaml")).read()
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    ckpt = os.path.join(out, "checkpoints", "day.json")
    state = json.load(open(ckpt)) if os.path.exists(ckpt) else {"day": 0, "kg": 0}
    with open(os.path.join(out, "starts.txt"), "a") as fh:
        fh.write(f"{state['day'] + 1}\\n")
    if os.environ.get("FAKE_HANG"):
        time.sleep(60)
    for day in range(state["day"] + 1, days + 1):
        time.sleep(float(os.environ.get("FAKE_DAY_SECONDS", "0")))
        state = {"day": day, "kg": state["kg"] + day * len(data)}
        json.dump(state, open(ckpt + ".tmp", "w"))
        os.replace(ckpt + ".tmp", ckpt)
    res = os.path.join(out, f"{days}days", "riomaior100_plastic", "emp", "run")
    os.makedirs(res, exist_ok=True)
    json.dump({"kg": state["kg"], "policy": policy, "license": os.environ.get("GRB_LICENSE_FILE")},
              open(os.path.join(res, "log_x_1N.json"), "w"))
    """
)


def _project(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "main.py").write_text(FAKE_MAIN)
    datasets = path / "data" / "simulator" / "datasets"
    datasets.mkdir(parents=True)
    (datasets / "d.npz").write_text("x" * 10)
    return path


def _job(tmp_path, days=6, **spec_fields):
    """A one-task remote job stored in a SQLite control plane file, with inputs staged."""
    source = _project(tmp_path / "submitter")
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "policy_x.yaml").write_text("x: 1\n")
    cell = CellSpec(cell_id="c/0", argv=["test_sim", f"sim.graph.n_days={days}"], policy_dir=str(policy))
    spec = JobSpec(
        name="rt", cells=[cell], provider=Provider.GCP, compute=ComputeKind.BATCH, image="img", **spec_fields
    )
    cp_uri, store_uri = f"sqlite://{tmp_path / 'cp.db'}", f"file://{tmp_path / 'store'}"
    cp, store = control_plane_from_uri(cp_uri), store_from_uri(store_uri)
    spec.inputs_manifest = stage_job_inputs(store, spec, source, "job-1")
    task = TaskRecord(task_id="task-1", job_id="job-1", cell=cell)
    cp.create_job(JobRecord(job_id="job-1", spec=spec), [task])
    return cp, store, cp_uri, store_uri


def _machine(tmp_path, name):
    """A fresh machine: an empty project root with only main.py, and an empty work directory."""
    root = tmp_path / name / "app"
    root.mkdir(parents=True)
    (root / "main.py").write_text(FAKE_MAIN)
    return root, tmp_path / name / "work"


@pytest.mark.unit
def test_profiles_resolve_local_uris_and_name_missing_providers(tmp_path):
    assert isinstance(control_plane_from_uri(f"sqlite://{tmp_path / 'a.db'}"), SQLiteControlPlane)
    assert isinstance(store_from_uri(f"file://{tmp_path / 's'}"), FilesystemObjectStore)
    with pytest.raises(ValueError, match="providers.gcp.storage"):
        store_from_uri("gs://bucket/prefix")
    with pytest.raises(ValueError, match="unsupported"):
        store_from_uri("ftp://x")


@pytest.mark.unit
def test_staging_round_trip_is_content_addressed_and_checked(tmp_path):
    source = _project(tmp_path / "src")
    policy = tmp_path / "pol"
    policy.mkdir()
    (policy / "policy_a.yaml").write_text("a: 1\n")
    store = FilesystemObjectStore(tmp_path / "store")
    spec = JobSpec(name="s", cells=[CellSpec(cell_id="c", argv=["test_sim"], policy_dir=str(policy))])
    key = stage_job_inputs(store, spec, source, "job-a")
    before = set(store.list("cas/"))
    stage_job_inputs(store, spec, source, "job-b")
    assert set(store.list("cas/")) == before  # unchanged inputs are not uploaded again
    root, work = tmp_path / "m" / "app", tmp_path / "m" / "work"
    got = fetch_inputs(store, key, "c", root, work)
    assert (root / "data/simulator/datasets/d.npz").read_text() == "x" * 10
    assert (got / "policy_a.yaml").read_text() == "a: 1\n"
    (root / "data/simulator/datasets/d.npz").write_text("tampered")
    blob = next(k for k in store.list("cas/") if (tmp_path / "store" / k).read_text() == "x" * 10)
    (tmp_path / "store" / blob).write_text("corrupt")
    with pytest.raises(ValueError, match="does not match"):
        fetch_inputs(store, key, "c", root, work)
    with pytest.raises(FileNotFoundError):
        stage_job_inputs(store, JobSpec(name="s", cells=spec.cells, inputs=["missing"]), source, "job-c")


def _start_runtime(tmp_path, cp_uri, store_uri, root, work, day_seconds):
    env = dict(
        os.environ,
        PYTHONPATH=str(REPO),
        WSR_CLOUD_CONTROL_PLANE=cp_uri,
        WSR_CLOUD_STORE=store_uri,
        WSR_CLOUD_JOB_ID="job-1",
        WSR_CLOUD_TASK_ID="task-1",
        WSR_CLOUD_WORKDIR=str(work),
        WSR_CLOUD_PROJECT_ROOT=str(root),
        WSR_CLOUD_SYNC_SECONDS="0.1",
        FAKE_DAY_SECONDS=str(day_seconds),
    )
    return subprocess.Popen([sys.executable, "-m", "logic.controllers.cloud.runtime"], env=env, cwd=REPO)


@pytest.mark.unit
def test_preempted_task_resumes_on_another_machine_with_identical_results(tmp_path):
    reference_cp, reference_store, *_ = _job(tmp_path / "reference")
    root, work = _machine(tmp_path / "reference", "m")
    assert run_task(reference_cp, reference_store, "job-1", "task-1", work, project_root=root, sync_seconds=0.05) == 0
    expected = json.loads(next((work / "outputs").rglob("log_x_1N.json")).read_text())

    cp, store, cp_uri, store_uri = _job(tmp_path / "preempted")
    root1, work1 = _machine(tmp_path / "preempted", "first")
    proc = _start_runtime(tmp_path, cp_uri, store_uri, root1, work1, day_seconds=0.4)
    deadline = time.time() + 30
    while True:
        assert time.time() < deadline and proc.poll() is None
        mirrored = [k for k in store.list("work/job-1/task-1/outputs/checkpoints/")]
        if mirrored and json.loads((tmp_path / "preempted" / "store" / mirrored[0]).read_text())["day"] >= 2:
            break
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == EXIT_TERMINATED
    assert "terminated" in cp.get_task("task-1").message

    root2, work2 = _machine(tmp_path / "preempted", "second")
    proc = _start_runtime(tmp_path, cp_uri, store_uri, root2, work2, day_seconds=0.0)
    assert proc.wait(timeout=60) == 0
    starts = (work2 / "outputs" / "starts.txt").read_text().split()
    assert starts[0] == "1" and int(starts[1]) >= 3, starts  # the second machine resumed after day 2
    result = json.loads(next((work2 / "outputs").rglob("log_x_1N.json")).read_text())
    assert result["kg"] == expected["kg"] and result["policy"] == expected["policy"]
    index = {f.relpath: f for f in cp.list_results("task-1")}
    assert "6days/riomaior100_plastic/emp/run/log_x_1N.json" in index
    assert "checkpoints/day.json" in index


@pytest.mark.unit
def test_idle_task_is_stopped(tmp_path, monkeypatch):
    cp, store, *_ = _job(tmp_path, resources=Resources(idle_seconds=1.0))
    root, work = _machine(tmp_path, "m")
    monkeypatch.setenv("FAKE_HANG", "1")
    started = time.time()
    assert run_task(cp, store, "job-1", "task-1", work, project_root=root, sync_seconds=0.1) == EXIT_IDLE
    assert time.time() - started < 20
    assert "no new output" in cp.get_task("task-1").message


@pytest.mark.unit
def test_gurobi_license_file_or_early_failure(tmp_path):
    cp, store, *_ = _job(tmp_path / "missing", gurobi=GurobiMode.WLS_SECRET, gurobi_secret="wls")
    cp.update_task("task-1", cell=CellSpec(cell_id="c/0", argv=["test_sim", "sim.graph.n_days=1"], needs_gurobi=True))
    root, work = _machine(tmp_path / "missing", "m")
    assert run_task(cp, store, "job-1", "task-1", work, project_root=root, license_text=None) == EXIT_NO_LICENSE

    cp, store, *_ = _job(tmp_path / "given", days=1, gurobi=GurobiMode.BYO, gurobi_secret="mine")
    root, work = _machine(tmp_path / "given", "m")
    assert (
        run_task(
            cp,
            store,
            "job-1",
            "task-1",
            work,
            project_root=root,
            sync_seconds=0.05,
            license_text="WLSACCESSID=a\nWLSSECRET=b\nLICENSEID=1",
        )
        == 0
    )
    license_path = Path(json.loads(next((work / "outputs").rglob("log_x_1N.json")).read_text())["license"])
    assert license_path.read_text().startswith("WLSACCESSID=a")
    assert stat.S_IMODE(license_path.stat().st_mode) == 0o600
    assert not any("WLSSECRET" in f.read_text(errors="ignore") for f in store.root.rglob("*") if f.is_file())


@pytest.mark.unit
def test_remote_jobs_stage_inputs_and_launch_the_runtime(tmp_path):
    source = _project(tmp_path / "src")
    backend = ScriptedBackend(provider=Provider.GCP, kind=ComputeKind.BATCH, polls_to_finish=0)
    service = CloudService(
        control_plane=SQLiteControlPlane(":memory:"),
        store=MemoryObjectStore(),
        backends={(Provider.GCP, ComputeKind.BATCH): backend},
        workroot=tmp_path / "work",
        clock=FakeClock(),
        remote=RemoteConfig(control_plane_uri="supabase://cp", store_uri="gs://b/p", project_root=source),
    )
    cell = CellSpec(cell_id="c", argv=["test_sim"])
    spec = JobSpec(
        name="r",
        cells=[cell],
        provider=Provider.GCP,
        compute=ComputeKind.BATCH,
        image="img@sha256:1",
        gurobi=GurobiMode.WLS_SECRET,
        gurobi_secret="gurobi-wls",
    )
    job = service.submit(spec)
    assert service.cp.get_job(job.job_id).spec.inputs_manifest == f"inputs/{job.job_id}/manifest.json"
    service.tick()
    launch = next(iter(backend.attempts.values())).launch
    assert launch.argv == ["python", "-m", "logic.controllers.cloud.runtime"] and launch.image == "img@sha256:1"
    assert launch.env["WSR_CLOUD_STORE"] == "gs://b/p" and launch.env["WSR_CLOUD_TASK_ID"]
    assert launch.secret_env == {"WSR_GUROBI_LICENSE": "gurobi-wls"} and launch.workdir is None
    with pytest.raises(ValueError, match="image"):
        service.submit(JobSpec(name="r", cells=[cell], provider=Provider.GCP, compute=ComputeKind.BATCH))
