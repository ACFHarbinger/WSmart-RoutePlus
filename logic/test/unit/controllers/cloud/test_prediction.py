"""Runtime and memory prediction (CR-11): features, the bundled model, machine factors, accuracy and submit."""

import pytest
from logic.controllers.cloud.local import SQLiteControlPlane
from logic.controllers.cloud.model import (
    CellSpec,
    ComputeKind,
    JobRecord,
    JobSpec,
    Provider,
    Resources,
    TaskRecord,
    TaskState,
)
from logic.controllers.cloud.prediction import (
    RuntimePredictor,
    cell_features,
    default_predictor,
    fitted_predictor,
    load_table,
    prediction_accuracy,
)
from logic.controllers.cloud.prediction.model import heuristic_memory_gb
from logic.controllers.cloud.service import CloudService
from logic.controllers.cloud.testing import FakeClock, MemoryObjectStore

ARGV = [
    "test_sim",
    "sim.graph.area=figueiradafoz",
    "sim.graph.num_loc=350",
    "sim.data_distribution=emp",
    "sim.graph.n_days=30",
    "sim.run_name=la_cls",
    "sim.policies=[alns]",
]


def _cell(cell_id="30d/figueiradafoz/350/emp/la_cls/lookahead/alns", argv=ARGV, **kw):
    return CellSpec(cell_id=cell_id, argv=list(argv), **kw)


@pytest.mark.unit
def test_features_come_from_argv_and_rerun_cell_ids():
    f = cell_features(_cell(), "e2-standard-4")
    assert f == {
        "policy": "alns",
        "packing": "lookahead",
        "improver": "cls",
        "distribution": "emp",
        "area": "figueiradafoz",
        "size": 350,
        "horizon": 30,
        "machine_type": "e2-standard-4",
    }
    only_id = cell_features(CellSpec(cell_id="90d/riomaior/170/gamma3/sl_ftsp/sl2/bpc", argv=["test_sim"]))
    assert (only_id["policy"], only_id["size"], only_id["horizon"], only_id["improver"]) == ("bpc", 170, 90, "ftsp")
    bare = cell_features(CellSpec(cell_id="c", argv=["test_sim"]))
    assert bare["size"] is None and bare["policy"] == "unknown" and bare["machine_type"] == "unknown"


@pytest.mark.unit
def test_bundled_table_and_held_out_error():
    rows = load_table()
    assert len(rows) == 810 and sum(r["source"] == "rerun_wall" for r in rows) == 210
    report = default_predictor().evaluate()
    assert report["holdout"] >= 50
    assert report["median_ape"] < 0.35 and report["p10_p90_coverage"] > 0.6  # 2026-10-20: 0.20 and 0.77


@pytest.mark.unit
def test_predictions_are_ordered_and_fall_back():
    cells = [_cell(), _cell("x", ["test_sim"], expected_wall_seconds=100.0), _cell("y", ["test_sim"])]
    model, expected, none = default_predictor().predict(cells)
    assert model.source == "model" and model.p10 <= model.p50 <= model.p90 and model.p50 > 0
    assert (expected.source, expected.p10, expected.p50, expected.p90) == ("expected", 50.0, 100.0, 200.0)
    assert none.source == "none" and none.p50 is None
    assert model.memory_gb == heuristic_memory_gb(350) and expected.memory_gb == heuristic_memory_gb(None)


@pytest.mark.unit
def test_without_scikit_learn_predictions_use_the_manifest(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_sklearn(name, *args, **kwargs):
        if name.startswith("sklearn"):
            raise ImportError("no sklearn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sklearn)
    predictor = RuntimePredictor(load_table()[:20])
    assert predictor.models is None and "no sklearn" in predictor.error
    (p,) = predictor.predict([_cell(expected_wall_seconds=60.0)])
    assert p.source == "expected" and p.p50 == 60.0


def _succeed(cp, task_id, started, finished):
    """Walk a queued task through the legal transitions to SUCCEEDED."""
    cp.update_task(task_id, state=TaskState.PLACING)
    cp.update_task(task_id, state=TaskState.STARTING)
    cp.update_task(task_id, state=TaskState.RUNNING, started_at=started)
    cp.update_task(task_id, state=TaskState.SUCCEEDED, finished_at=finished)


def _finished_job(cp, job_id, cells, machine, walls, peaks_mb=None):
    spec = JobSpec(name=job_id, cells=cells, resources=Resources(machine_type=machine))
    tasks = [TaskRecord(task_id=f"{job_id}-{i}", job_id=job_id, cell=c) for i, c in enumerate(cells)]
    cp.create_job(JobRecord(job_id=job_id, spec=spec), tasks)
    for i, (task, wall) in enumerate(zip(tasks, walls, strict=True)):
        _succeed(cp, task.task_id, 1000.0, 1000.0 + wall)
        if peaks_mb:
            cp.put_metrics(task.task_id, [{"at": 1001.0, "kind": "resource", "memory_rss_mb": peaks_mb[i]}])
    return tasks


@pytest.mark.unit
def test_machine_factor_and_memory_come_from_completed_tasks():
    cp = SQLiteControlPlane(":memory:")
    laptop = default_predictor().predict([_cell()])[0].p50
    cells = [_cell(f"30d/figueiradafoz/350/emp/la_cls/lookahead/alns#{i}") for i in range(3)]
    _finished_job(cp, "job-a", cells, "slow-vm", [2 * laptop] * 3, peaks_mb=[1000.0, 2000.0, 4000.0])
    predictor = fitted_predictor(cp)
    assert predictor.machine_factors["slow-vm"] == pytest.approx(2.0, rel=0.01)
    (p,) = predictor.predict([_cell()], "slow-vm")
    assert p.p50 == pytest.approx(2 * laptop, rel=0.01) and p.machine_factor == pytest.approx(2.0, rel=0.01)
    assert p.memory_source == "observed" and p.memory_gb == pytest.approx(1.25 * 4000 / 1024, abs=0.01)
    (other,) = predictor.predict([_cell()], "fast-vm")
    assert other.machine_factor == 1.0


@pytest.mark.unit
def test_submit_fills_estimates_and_accuracy_is_tracked(tmp_path):
    clock = FakeClock()
    cp = SQLiteControlPlane(":memory:")
    from logic.controllers.cloud.testing import ScriptedBackend

    service = CloudService(
        control_plane=cp,
        store=MemoryObjectStore(),
        backends={
            (Provider.LOCAL, ComputeKind.PROCESS): ScriptedBackend(provider=Provider.LOCAL, kind=ComputeKind.PROCESS)
        },
        workroot=tmp_path,
        clock=clock,
        predictor_factory=default_predictor,
    )
    given = _cell("given", ["test_sim"], expected_wall_seconds=42.0)
    job = service.submit(JobSpec(name="p", cells=[_cell(), given]))
    tasks = {t.cell.cell_id: t for t in cp.list_tasks(job.job_id)}
    predicted = tasks["30d/figueiradafoz/350/emp/la_cls/lookahead/alns"].cell.expected_wall_seconds
    assert predicted and predicted > 0 and tasks["given"].cell.expected_wall_seconds == 42.0  # never overwritten
    event = [e for e in cp.list_events(job.job_id) if e.kind == "prediction"][0]
    assert set(event.data["cells"]) == {"30d/figueiradafoz/350/emp/la_cls/lookahead/alns", "given"}

    for task in tasks.values():
        _succeed(cp, task.task_id, 0.0, predicted)
    report = prediction_accuracy(cp, job.job_id)
    rows = {r["cell_id"]: r for r in report["tasks"]}
    assert rows["30d/figueiradafoz/350/emp/la_cls/lookahead/alns"]["ratio"] == 1.0
    assert rows["30d/figueiradafoz/350/emp/la_cls/lookahead/alns"]["inside"] is True
    assert report["median_ape"] is not None and report["coverage"] is not None

    from logic.controllers.cloud import cli

    ctx = cli._Context({"predict_command": "accuracy", "job_id": job.job_id, "json": True}, tmp_path, tmp_path, service)
    assert cli._COMMANDS["predict"](ctx)["job_id"] == job.job_id
    command = "test_sim " + " ".join(ARGV[1:])
    ctx = cli._Context(
        {"predict_command": "estimate", "sim_command": command, "json": True}, tmp_path, tmp_path, service
    )
    payload = cli._COMMANDS["predict"](ctx)
    assert payload["cells"][0]["source"] == "model" and payload["total_p50_seconds"] > 0


@pytest.mark.unit
def test_no_predictor_leaves_specs_unchanged(tmp_path):
    from logic.controllers.cloud.testing import ScriptedBackend

    service = CloudService(
        control_plane=SQLiteControlPlane(":memory:"),
        store=MemoryObjectStore(),
        backends={
            (Provider.LOCAL, ComputeKind.PROCESS): ScriptedBackend(provider=Provider.LOCAL, kind=ComputeKind.PROCESS)
        },
        workroot=tmp_path,
        clock=FakeClock(),
    )
    job = service.submit(JobSpec(name="p", cells=[_cell()]))
    assert service.cp.list_tasks(job.job_id)[0].cell.expected_wall_seconds is None
    assert not [e for e in service.cp.list_events(job.job_id) if e.kind == "prediction"]
    assert service.predict(JobSpec(name="p", cells=[_cell()])) is None
