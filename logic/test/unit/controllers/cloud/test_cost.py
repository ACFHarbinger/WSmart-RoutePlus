"""Cost (CR-10): price table, estimates, budget caps at submit, accrual, stops and billing imports."""

import datetime as dt

import pytest
from logic.controllers.cloud.cost import (
    BudgetExceeded,
    BudgetPolicy,
    PriceTable,
    estimate_job,
    import_billing_csv,
)
from logic.controllers.cloud.local import SQLiteControlPlane
from logic.controllers.cloud.model import CellSpec, ComputeKind, JobSpec, Provider, Resources, TaskState
from logic.controllers.cloud.prediction import Prediction
from logic.controllers.cloud.service import CloudService, RemoteConfig
from logic.controllers.cloud.testing import FakeClock, MemoryObjectStore, ScriptedBackend

# 2026-10-15 12:00 UTC, so a month boundary is not crossed while a test advances the clock.
NOW = dt.datetime(2026, 10, 15, 12, tzinfo=dt.timezone.utc).timestamp()


class _Predictor:
    """Every cell takes p10/p50/p90 = 0.5/1/2 x ``seconds`` (default one hour)."""

    def __init__(self, seconds=3600.0):
        self.seconds = seconds

    def predict(self, cells, machine_type=None):
        s = self.seconds
        return [Prediction(c.cell_id, 0.5 * s, s, 2 * s, 2.0, "model", "heuristic") for c in cells]


def _service(tmp_path, budget=None, rate=2.0, polls_to_finish=1000, predicted_seconds=3600.0):
    clock = FakeClock(NOW)
    (tmp_path / "data" / "simulator").mkdir(parents=True, exist_ok=True)
    prices = PriceTable(tmp_path / "prices.json")
    prices.overrides = {"aws": {"*": {"m1": [rate, rate / 2]}}}
    backend = ScriptedBackend(polls_to_finish=polls_to_finish, provider=Provider.AWS, kind=ComputeKind.BATCH)
    service = CloudService(
        control_plane=SQLiteControlPlane(":memory:"),
        store=MemoryObjectStore(),
        backends={(Provider.AWS, ComputeKind.BATCH): backend},
        workroot=tmp_path / "work",
        clock=clock,
        remote=RemoteConfig(
            control_plane_uri="sqlite:///cp.db", store_uri=f"file://{tmp_path}/store", project_root=tmp_path
        ),
        predictor_factory=lambda: _Predictor(predicted_seconds),
        prices=prices,
        budget=budget,
    )
    return service, clock


def _spec(cells=1, budget=None, spot=False):
    return JobSpec(
        name="cost",
        cells=[CellSpec(cell_id=f"c{i}", argv=["test_sim"]) for i in range(cells)],
        provider=Provider.AWS,
        compute=ComputeKind.BATCH,
        resources=Resources(machine_type="m1", spot=spot, budget_usd=budget),
        image="release:latest",
        max_parallel=cells,
    )


@pytest.mark.unit
def test_price_table_overrides_static_tables_and_updates(tmp_path):
    table = PriceTable(tmp_path / "prices.json", clock=lambda: 99.0)
    assert table.price_per_hour("aws", "c6i.large", "us-east-1", False) == 0.085  # the provider's static table
    assert table.price_per_hour("aws", "nope", None, False) is None
    assert table.price_per_hour("nowhere", "x", None, False) is None
    stored = table.update("aws", "eu-west-1", lambda provider, region: {"c6i.large": (0.2, 0.05)})
    assert stored == 1 and table.price_per_hour("aws", "c6i.large", "eu-west-1", True) == 0.05
    assert table.price_per_hour("aws", "c6i.large", "us-east-1", False) == 0.085  # other regions untouched
    reloaded = PriceTable(tmp_path / "prices.json")
    assert reloaded.price_per_hour("aws", "c6i.large", "eu-west-1", False) == 0.2
    assert reloaded.updated_at == {"aws/eu-west-1": 99.0}


@pytest.mark.unit
def test_estimate_sums_cells_and_reports_unknowns():
    preds = [
        Prediction("a", 1800.0, 3600.0, 7200.0, 2.0, "model", "heuristic"),
        Prediction("b", None, None, None, 2.0, "none", "heuristic"),
    ]
    est = estimate_job(preds, 2.0, False, "m1")
    assert (est.low, est.mid, est.high) == (1.0, 2.0, 4.0) and est.unpredicted_cells == ["b"]
    unknown = estimate_job(preds, None, False, "m1")
    assert unknown.mid is None and unknown.cells[0]["mid"] is None


@pytest.mark.unit
def test_submit_refuses_a_job_over_its_cap_and_warns_on_p90(tmp_path):
    service, _ = _service(tmp_path)
    with pytest.raises(BudgetExceeded, match="job budget"):
        service.submit(_spec(cells=2, budget=3.0))  # p50: 2 cells x 1 h x 2 USD/h = 4 USD
    assert service.cp.list_jobs() == []  # nothing stored
    job = service.submit(_spec(cells=1, budget=3.0))  # p50 2 USD fits, p90 4 USD does not
    events = service.cp.list_events(job.job_id)
    estimate = [e.data for e in events if e.kind == "cost_estimate"][0]
    assert (estimate["mid"], estimate["high"], estimate["price_per_hour"]) == (2.0, 4.0, 2.0)
    assert [e.data["estimate"] for e in events if e.kind == "budget_warning"] == [True]
    assert service.submit(_spec(spot=True, budget=1.0))  # spot: p50 1 x 1 USD = 1 USD fits exactly


@pytest.mark.unit
def test_running_job_accrues_warns_and_is_stopped_at_its_cap(tmp_path):
    # Predicted 10 s at 1 USD per second: 10 USD estimated, under the 100 USD cap; the task then runs long.
    service, clock = _service(tmp_path, rate=3600.0, predicted_seconds=10.0)
    job = service.submit(_spec(cells=1, budget=100.0))
    for _ in range(12):
        service.tick()
        clock.advance(30.0)
    task = service.cp.list_tasks(job.job_id)[0]
    assert task.state == TaskState.CANCELLED
    costs = [r for r in service.cp.get_metrics(task.task_id) if r["kind"] == "cost"]
    assert costs and costs[-1]["rate_usd_h"] == 3600.0
    assert all(b["at"] - a["at"] >= 60.0 for a, b in zip(costs[:-2], costs[1:-1]))  # rate-limited while running
    events = service.cp.list_events(job.job_id)
    assert sorted(e.data["fraction"] for e in events if e.kind == "budget_warning") == [0.5, 0.8, 0.95]
    stops = [e.data for e in events if e.kind == "budget_stop"]
    assert len(stops) == 1 and stops[0]["scope"] == "job" and stops[0]["accrued_usd"] >= 100.0
    assert service.job_accrued(service.cp.get_job(job.job_id)) < 100.0 + 31.0  # stopped within one pass of the cap


@pytest.mark.unit
def test_monthly_cap_refuses_new_jobs_and_stops_running_ones(tmp_path):
    service, clock = _service(tmp_path, rate=3600.0, budget=BudgetPolicy(monthly_usd=7300.0))
    first = service.submit(_spec())  # p50: 3,600 USD of the 7,300
    second = service.submit(_spec())  # 7,200 of the 7,300
    with pytest.raises(BudgetExceeded, match="month budget"):
        service.submit(_spec())
    for _ in range(4):
        service.tick()
        clock.advance(30.0)
    assert service.month_accrued(clock()) > 0
    service.budget.monthly_usd = 100.0  # the month is now over its cap: both running jobs stop
    for _ in range(3):
        service.tick()
        clock.advance(30.0)
    for job in (first, second):
        assert service.cp.list_tasks(job.job_id)[0].state == TaskState.CANCELLED
        assert [e.data["scope"] for e in service.cp.list_events(job.job_id) if e.kind == "budget_stop"] == ["month"]


@pytest.mark.unit
def test_local_jobs_cost_nothing(tmp_path):
    clock = FakeClock(NOW)
    service = CloudService(
        control_plane=SQLiteControlPlane(":memory:"),
        store=MemoryObjectStore(),
        backends={(Provider.LOCAL, ComputeKind.PROCESS): ScriptedBackend()},
        workroot=tmp_path,
        clock=clock,
        predictor_factory=_Predictor,
    )
    job = service.submit(
        JobSpec(name="l", cells=[CellSpec(cell_id="c", argv=["test_sim"])], resources=Resources(budget_usd=0.0))
    )
    estimate = [e.data for e in service.cp.list_events(job.job_id) if e.kind == "cost_estimate"][0]
    assert estimate["mid"] == 0.0


@pytest.mark.unit
def test_billing_export_is_attributed_to_jobs(tmp_path):
    service, clock = _service(tmp_path)
    job = service.submit(_spec())
    export = tmp_path / "billing.csv"
    export.write_text(
        "service,labels.wsr_job_id,cost\n"
        f"Compute,{job.job_id},1.25\n"
        f"Storage,{job.job_id},0.05\n"
        "Compute,job-unknown,9.0\n"
        "Network,,3.0\n"
    )
    assert import_billing_csv(service.cp, export, "gcp", clock()) == {job.job_id: 1.3}
    billing = [e.data for e in service.cp.list_events(job.job_id) if e.kind == "billing"]
    assert billing == [{"provider": "gcp", "cost_usd": 1.3, "source": str(export)}]
    (tmp_path / "bad.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="job label"):
        import_billing_csv(service.cp, tmp_path / "bad.csv", "gcp", clock())


@pytest.mark.unit
def test_billing_export_reads_thousands_separators_and_rejects_bad_costs(tmp_path):
    service, clock = _service(tmp_path)
    job = service.submit(_spec())
    export = tmp_path / "billing.csv"
    export.write_text(f'service,labels.wsr_job_id,cost\nCompute,{job.job_id},"1,234.50"\nStorage,{job.job_id},0.25\n')
    assert import_billing_csv(service.cp, export, "gcp", clock()) == {job.job_id: 1234.75}
    for bad in ("n/a", '"1234,56"'):
        export.write_text(f"service,labels.wsr_job_id,cost\nCompute,{job.job_id},{bad}\n")
        with pytest.raises(ValueError, match="unparseable cost"):
            import_billing_csv(service.cp, export, "gcp", clock())


@pytest.mark.unit
def test_cli_cost_commands(tmp_path):
    from logic.controllers.cloud import cli

    service, clock = _service(tmp_path)
    job = service.submit(_spec())
    show = cli._COMMANDS["cost"](
        cli._Context({"cost_command": "show", "job_id": job.job_id, "json": True}, tmp_path, tmp_path, service)
    )
    assert show["estimate"]["mid"] == 2.0 and show["accrued_usd"] == 0.0 and len(show["tasks"]) == 1
    prices = cli._COMMANDS["cost"](
        cli._Context(
            {"cost_command": "prices", "provider": "aws", "machine_type": "m1", "json": True},
            tmp_path,
            tmp_path,
            service,
        )
    )
    assert (prices["on_demand_usd_h"], prices["spot_usd_h"]) == (2.0, 1.0)
    opts = {
        "cost_command": "estimate",
        "sim_command": "test_sim sim.graph.n_days=1",
        "provider": "aws",
        "compute": "batch",
        "machine_type": "m1",
        "json": True,
    }
    estimate = cli._COMMANDS["cost"](cli._Context(opts, tmp_path, tmp_path, service))
    assert (estimate["mid"], estimate["cost_usd"], estimate["eta_seconds"], estimate["memory_required_gb"]) == (
        2.0,
        2.0,
        3600.0,
        2.0,
    )
    from logic.controllers.cloud.cli import _makespan

    assert _makespan([5.0, 4.0, 3.0, 3.0], 2) == 8.0 and _makespan([], 3) is None
