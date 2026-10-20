"""Scheduler (CR-09): placement and spill-over, health, priorities, auto-scaling, packing and the warm pool."""

import datetime as dt

import pytest
from logic.controllers.cloud.cost import PriceTable
from logic.controllers.cloud.interfaces import CapacityError
from logic.controllers.cloud.local import SQLiteControlPlane
from logic.controllers.cloud.model import (
    ACTIVE,
    CellSpec,
    ComputeKind,
    JobSpec,
    PlacementTarget,
    Provider,
    Resources,
    TaskState,
)
from logic.controllers.cloud.prediction import Prediction
from logic.controllers.cloud.scheduler import (
    PoolPolicy,
    TargetHealth,
    pack,
    rank_targets,
    run_pool_worker,
    slots_per_machine,
)
from logic.controllers.cloud.service import CloudService, RemoteConfig
from logic.controllers.cloud.testing import FakeClock, MemoryObjectStore, ScriptedBackend

NOW = dt.datetime(2026, 10, 15, 12, tzinfo=dt.timezone.utc).timestamp()
GCP = PlacementTarget(Provider.GCP, ComputeKind.BATCH, machine_type="m", spot=True)
AWS = PlacementTarget(Provider.AWS, ComputeKind.BATCH, machine_type="m")
AZURE = PlacementTarget(Provider.AZURE, ComputeKind.VM, machine_type="m", max_tasks=2)
TERMINAL = (TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED)


class _Predictor:
    def __init__(self, seconds=3600.0, memory_gb=2.0):
        self.seconds, self.memory_gb = seconds, memory_gb

    def predict(self, cells, machine_type=None):
        s = self.seconds
        return [Prediction(c.cell_id, s / 2, s, 2 * s, self.memory_gb, "model", "heuristic") for c in cells]


def _prices(tmp_path):
    table = PriceTable(tmp_path / "prices.json")
    table.overrides = {p: {"*": {"m": [rate, rate / 2]}} for p, rate in (("gcp", 0.2), ("aws", 0.2), ("azure", 0.3))}
    return table


def _service(tmp_path, backends, predictor=None, **kw):
    (tmp_path / "data" / "simulator").mkdir(parents=True, exist_ok=True)
    clock = FakeClock(NOW)
    return (
        CloudService(
            control_plane=kw.pop("cp", None) or SQLiteControlPlane(":memory:"),
            store=MemoryObjectStore(),
            backends=backends,
            workroot=tmp_path / "work",
            clock=clock,
            remote=RemoteConfig(
                control_plane_uri="sqlite:///cp.db", store_uri=f"file://{tmp_path}/s", project_root=tmp_path
            ),
            predictor_factory=(lambda: predictor) if predictor else None,
            prices=_prices(tmp_path),
            **kw,
        ),
        clock,
    )


def _spec(cells, placement=(), **kw):
    resources = kw.pop("resources", Resources(machine_type="m", max_retries=3))
    return JobSpec(
        name=kw.pop("name", "s"),
        cells=[CellSpec(cell_id=f"c{i}", argv=["test_sim"]) for i in range(cells)],
        provider=Provider.GCP,
        compute=ComputeKind.BATCH,
        resources=resources,
        image="release:latest",
        placement=list(placement),
        **kw,
    )


def _drain(service, clock, limit=400, watch=None):
    for _ in range(limit):
        service.tick()
        if watch:
            watch()
        clock.advance(30.0)
        jobs = service.cp.list_jobs()
        if all(t.state in TERMINAL for j in jobs for t in service.cp.list_tasks(j.job_id)):
            return
    raise AssertionError("queue did not drain")


@pytest.mark.unit
def test_ranking_prefers_cheap_healthy_targets_under_quota():
    health = TargetHealth(failure_threshold=2, cooldown_seconds=100.0)
    spec = _spec(1, [AZURE, AWS, GCP])
    served = {(Provider.GCP, ComputeKind.BATCH), (Provider.AWS, ComputeKind.BATCH), (Provider.AZURE, ComputeKind.VM)}
    prices = {"gcp": 0.1, "aws": 0.2, "azure": 0.3}

    def price(view):
        return prices[view.provider.value]

    keys = lambda ranked: [t.provider.value for t in ranked]  # noqa: E731
    assert keys(rank_targets(spec, served, {}, {}, health, 0.0, price, 3600.0)) == ["gcp", "aws", "azure"]
    for outcome in ["preempted"] * 4 + ["succeeded"] * 4:  # half the spot attempts preempted: 0.1 x 2 > 0.2 on demand
        health.record(GCP.key(), outcome, 0.0)
    assert keys(rank_targets(spec, served, {}, {}, health, 0.0, price, 3600.0)) == ["aws", "gcp", "azure"]
    assert "azure" not in keys(rank_targets(spec, served, {AZURE.key(): 2}, {}, health, 0.0, price, 3600.0))  # quota
    health.record(AWS.key(), "capacity", 1.0)
    health.record(AWS.key(), "launch_failed", 2.0)  # circuit opens
    assert "aws" not in keys(rank_targets(spec, served, {}, {}, health, 50.0, price, 3600.0))
    assert "aws" in keys(rank_targets(spec, served, {}, {}, health, 103.0, price, 3600.0))  # half-open after cooldown
    unpriced = rank_targets(
        spec, served, {}, {}, TargetHealth(), 0.0, lambda v: None if v.provider.value == "gcp" else 1.0, None
    )
    assert keys(unpriced)[-1] == "gcp"


@pytest.mark.unit
def test_packing_first_fit_decreasing():
    items = [("a", 1, 6.0), ("b", 1, 6.0), ("c", 1, 6.0), ("d", 1, 3.0), ("e", 1, 20.0)]
    bins = pack(items, cpus=4, memory_gb=16.0)
    assert sorted(map(sorted, bins)) == [["a", "b", "d"], ["c"], ["e"]]
    assert (
        slots_per_machine(4, 16.0, 1, 6.0) == 2
        and slots_per_machine(4, 16.0, 2, 1.0) == 2
        and slots_per_machine(1, 1, 2, 9) == 1
    )


class _Flaky(ScriptedBackend):
    """A scripted backend with a running-task capacity and the first ``broken`` submits raising."""

    def __init__(self, capacity=None, broken=0, **kw):
        super().__init__(**kw)
        self.capacity, self.broken, self.submits = capacity, broken, 0

    def running(self):
        return sum(1 for a in self.attempts.values() if not a.cancelled and a.polls <= self.polls_to_finish)

    def submit(self, launch):
        self.submits += 1
        if self.submits <= self.broken:
            raise RuntimeError("scripted launch failure")
        if self.capacity is not None and self.running() >= self.capacity:
            raise CapacityError("scripted: full")
        return super().submit(launch)


@pytest.mark.unit
def test_multi_provider_queue_drains_within_caps(tmp_path):
    attempts = {"n": 0}

    def gcp_script(launch):
        attempts["n"] += 1
        return "preempt" if attempts["n"] % 4 == 0 else "succeed"  # every fourth spot attempt is preempted

    gcp = _Flaky(capacity=2, script=gcp_script, polls_to_finish=2, provider=Provider.GCP, kind=ComputeKind.BATCH)
    aws = _Flaky(broken=3, polls_to_finish=2, provider=Provider.AWS, kind=ComputeKind.BATCH)
    azure = _Flaky(
        script=lambda launch: "fail" if launch.task.cell.cell_id == "c7" and launch.task.attempt == 1 else "succeed",
        polls_to_finish=2,
        provider=Provider.AZURE,
        kind=ComputeKind.VM,
    )
    backends = {
        (Provider.GCP, ComputeKind.BATCH): gcp,
        (Provider.AWS, ComputeKind.BATCH): aws,
        (Provider.AZURE, ComputeKind.VM): azure,
    }
    service, clock = _service(tmp_path, backends, predictor=_Predictor())
    job = service.submit(_spec(20, [GCP, AWS, AZURE], max_parallel=6))
    peaks = {"azure": 0, "gcp": 0, "parallel": 0}

    def watch():
        tasks = service.cp.list_tasks(job.job_id)
        active = [t for t in tasks if t.state in ACTIVE]
        peaks["parallel"] = max(peaks["parallel"], len(active))
        peaks["azure"] = max(peaks["azure"], sum(t.placement == AZURE.key() for t in active))
        peaks["gcp"] = max(peaks["gcp"], gcp.running())

    _drain(service, clock, watch=watch)
    tasks = service.cp.list_tasks(job.job_id)
    assert all(t.state == TaskState.SUCCEEDED for t in tasks), [(t.cell.cell_id, t.state, t.message) for t in tasks]
    assert peaks["azure"] <= 2 and peaks["gcp"] <= 2 and peaks["parallel"] <= 6  # quota, capacity, max_parallel
    events = service.cp.list_events(job.job_id)
    spilled = [e.data for e in events if e.kind == "placement" and e.data["skipped"]]
    assert spilled and any("launch failed" in s for e in spilled for s in e["skipped"])
    assert any(e.kind == "retry" for e in events)  # preemptions and the scripted failure were retried
    assert {t.placement for t in tasks} <= {GCP.key(), AWS.key(), AZURE.key()}


@pytest.mark.unit
def test_tasks_fail_with_a_reason_when_every_target_refuses(tmp_path):
    aws = _Flaky(broken=10**6, provider=Provider.AWS, kind=ComputeKind.BATCH)
    azure = _Flaky(broken=10**6, provider=Provider.AZURE, kind=ComputeKind.VM)
    service, clock = _service(
        tmp_path, {(Provider.AWS, ComputeKind.BATCH): aws, (Provider.AZURE, ComputeKind.VM): azure}
    )
    job = service.submit(_spec(2, [AWS, AZURE], max_parallel=2))
    _drain(service, clock)
    for task in service.cp.list_tasks(job.job_id):
        assert (
            task.state == TaskState.FAILED
            and task.message.startswith("launch failed:")
            and "scripted launch failure" in task.message
        )


@pytest.mark.unit
def test_higher_priority_job_takes_the_shared_quota_first(tmp_path):
    only = PlacementTarget(Provider.AWS, ComputeKind.BATCH, machine_type="m", max_tasks=1)
    aws = ScriptedBackend(polls_to_finish=3, provider=Provider.AWS, kind=ComputeKind.BATCH)
    service, clock = _service(tmp_path, {(Provider.AWS, ComputeKind.BATCH): aws})
    low = service.submit(_spec(2, [only], name="low", max_parallel=2))
    clock.advance(1.0)
    high = service.submit(_spec(2, [only], name="high", priority=5, max_parallel=2))
    order = []

    def watch():
        for job in (low, high):
            for t in service.cp.list_tasks(job.job_id):
                if t.state in ACTIVE and (job.spec.name, t.task_id) not in order:
                    order.append((job.spec.name, t.task_id))

    _drain(service, clock, watch=watch)
    assert [name for name, _ in order] == ["high", "high", "low", "low"]


@pytest.mark.unit
def test_budget_limits_parallelism_and_stops_when_exhausted(tmp_path):
    # 0.2 USD/h x 36 s = 0.002 USD a task: the six-task job (0.012 USD) fits a 0.0125 USD cap.
    aws = ScriptedBackend(polls_to_finish=1, provider=Provider.AWS, kind=ComputeKind.BATCH)
    service, clock = _service(tmp_path, {(Provider.AWS, ComputeKind.BATCH): aws}, predictor=_Predictor(seconds=36.0))
    job = service.submit(_spec(6, [AWS], max_parallel=3, resources=Resources(machine_type="m", budget_usd=0.0125)))
    peak = 0
    for _ in range(10):
        service.tick()
        peak = max(peak, sum(t.state in ACTIVE for t in service.cp.list_tasks(job.job_id)))
        clock.advance(36.0)
    tasks = service.cp.list_tasks(job.job_id)
    # The first three ran twice as long as predicted (0.004 USD each): the 0.0005 USD left cannot pay for a fourth.
    assert peak == 3 and sum(t.state == TaskState.SUCCEEDED for t in tasks) == 3
    assert {t.state for t in tasks} == {TaskState.SUCCEEDED, TaskState.CANCELLED}
    stops = [e.data for e in service.cp.list_events(job.job_id) if e.kind == "budget_stop"]
    assert [s["reason"] for s in stops] == ["exhausted"]


@pytest.mark.unit
def test_warm_pool_packs_workers_and_workers_drain_pool_jobs(tmp_path):
    cp = SQLiteControlPlane(":memory:")
    vm = PlacementTarget(Provider.AWS, ComputeKind.VM, machine_type="m")
    worker_backend = ScriptedBackend(script=lambda launch: "hang", provider=Provider.AWS, kind=ComputeKind.VM)
    pool = PoolPolicy(
        target=vm,
        cpus=4,
        memory_gb=16.0,
        max_workers=2,
        idle_seconds=60.0,
        heartbeat_timeout=120.0,
        image="release:latest",
    )
    coordinator, clock = _service(
        tmp_path,
        {(Provider.AWS, ComputeKind.VM): worker_backend, (Provider.AWS, ComputeKind.BATCH): ScriptedBackend()},
        predictor=_Predictor(memory_gb=6.0),
        cp=cp,
        pool=pool,
    )
    pooled = JobSpec(
        name="pooled",
        cells=[CellSpec(cell_id=f"p{i}", argv=["test_sim"]) for i in range(6)],
        provider=Provider.AWS,
        compute=ComputeKind.BATCH,
        resources=Resources(machine_type="m", cpus=1, max_retries=2),
        image="release:latest",
        pool=True,
        max_parallel=6,
    )
    job = coordinator.submit(pooled)
    coordinator.tick()
    assert all(t.state == TaskState.QUEUED for t in cp.list_tasks(job.job_id))  # the coordinator never starts them
    assert len(worker_backend.attempts) == 2  # 6 x 6 GB on 16 GB machines needs 3, capped at max_workers=2
    launch = next(iter(worker_backend.attempts.values())).launch
    assert launch.argv[-6:] == ["worker", "--pool", "--slots", "2", "--idle", "60.0"]
    assert launch.env["WSR_CLOUD_CONTROL_PLANE"] == "sqlite:///cp.db"
    coordinator.tick()
    assert len(worker_backend.attempts) == 2  # never more than the demand or the cap

    local = ScriptedBackend(polls_to_finish=1)
    worker, wclock = _service(
        tmp_path / "w", {(Provider.LOCAL, ComputeKind.PROCESS): local}, cp=cp, pool_mode=True, pool_slots=2
    )
    worker.owner = "worker-1"
    peak = {"n": 0}
    real_tick = worker.tick

    def counting_tick():
        changes = real_tick()
        running = [t for t in cp.list_tasks(job.job_id) if t.state in ACTIVE and t.lease_owner == "worker-1"]
        peak["n"] = max(peak["n"], len(running))
        wclock.advance(10.0)
        return changes

    worker.tick = counting_tick
    reason = run_pool_worker(
        worker, idle_seconds=30.0, poll_seconds=0.0, clock=wclock, sleep=lambda s: None, max_ticks=200
    )
    assert reason.startswith("idle") and peak["n"] == 2  # two slots, then scale to zero
    assert all(t.state == TaskState.SUCCEEDED for t in cp.list_tasks(job.job_id))


@pytest.mark.unit
def test_coordinator_requeues_tasks_of_a_lost_pool_worker(tmp_path):
    cp = SQLiteControlPlane(":memory:")
    pool = PoolPolicy(target=PlacementTarget(Provider.AWS, ComputeKind.VM), heartbeat_timeout=120.0)
    coordinator, clock = _service(
        tmp_path,
        {
            (Provider.AWS, ComputeKind.BATCH): ScriptedBackend(),
            (Provider.AWS, ComputeKind.VM): ScriptedBackend(script=lambda l: "hang"),
        },
        cp=cp,
        pool=pool,
    )
    job = coordinator.submit(
        JobSpec(
            name="p",
            cells=[CellSpec(cell_id="x", argv=["test_sim"])],
            provider=Provider.AWS,
            compute=ComputeKind.BATCH,
            image="img",
            pool=True,
            resources=Resources(max_retries=2),
        )
    )
    worker, _ = _service(
        tmp_path / "w",
        {(Provider.LOCAL, ComputeKind.PROCESS): ScriptedBackend(script=lambda l: "hang")},
        cp=cp,
        pool_mode=True,
    )
    worker.clock = clock
    worker.tick()  # the worker takes the task and goes silent
    task = cp.list_tasks(job.job_id)[0]
    assert task.state in (TaskState.STARTING, TaskState.RUNNING) and task.lease_owner == worker.owner
    clock.advance(121.0)
    coordinator.tick()
    task = cp.list_tasks(job.job_id)[0]
    assert task.state == TaskState.QUEUED and task.attempt == 2
    assert "pool worker lost" in [e for e in cp.list_events(job.job_id) if e.kind == "retry"][0].data["reason"]


@pytest.mark.unit
def test_targets_parse_from_the_cli_and_manifests(tmp_path):
    import json

    from logic.controllers.cloud.cli import parse_target
    from logic.controllers.cloud.manifest import load_manifest
    from logic.controllers.cloud.model import spec_from_dict, to_dict

    t = parse_target("gcp/batch/europe-west1/e2-standard-4/spot:8")
    assert (t.provider, t.compute, t.region, t.machine_type, t.spot, t.max_tasks) == (
        Provider.GCP,
        ComputeKind.BATCH,
        "europe-west1",
        "e2-standard-4",
        True,
        8,
    )
    assert (
        parse_target("aws/vm").key() == "aws/vm/-/-"
        and parse_target("azure/vm/-/Standard_D4s_v5").machine_type == "Standard_D4s_v5"
    )
    with pytest.raises(ValueError):
        parse_target("gcp")
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "m",
                "priority": 3,
                "pool": True,
                "placement": [{"provider": "aws", "compute": "batch", "max_tasks": 2}],
                "cells": [{"cell_id": "a", "command": "test_sim sim.graph.n_days=1"}],
            }
        )
    )
    spec = load_manifest(manifest)
    assert spec.priority == 3 and spec.pool and spec.placement[0].max_tasks == 2
    assert spec_from_dict(to_dict(spec)).placement == spec.placement  # round-trips through the control planes
