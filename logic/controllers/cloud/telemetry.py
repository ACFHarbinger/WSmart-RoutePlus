"""Task instrumentation: resource usage, simulation progress, events, and the tracking import.

The task runtime samples, every few seconds:

- **resource** records: CPU, resident memory and threads of the simulator's
  process tree, the work directory's size, and the machine's network counters
  (only when ``psutil`` is installed; without it no resource records are written);
- **progress** records: one per policy, sample and day, read from the
  simulator's realtime log (``log_realtime_*.jsonl``, written by
  ``tracking/logging/modules/gui.py``): the day, the horizon and the day's KPIs.

:class:`TelemetryBuffer` batches them into ``ControlPlane.put_metrics`` at most
once per flush interval, so a fast simulator cannot flood the control plane.
Cost records (``kind="cost"``) are written by the cost accounting (CR-10) in the
same stream.

:func:`import_job` copies a job's tasks, metrics and events from the control
plane into the local tracking database (``logic/src/tracking``): one tracking
run per task (``run_type="cloud_task"``), with the job, cell, provider and
attempt as tags, metrics keyed ``resource/<name>``, ``progress/<kpi>`` and
``cost/<name>``, and lifecycle events in the ``run_events`` table.

Attributes:
    TaskTelemetry: What the task runtime calls on every supervision tick.
    ResourceSampler: psutil sampling of a process tree.
    ProgressReader: Incremental reader of the simulator's realtime logs.
    TelemetryBuffer: Rate-limited batching of records into the control plane.
    import_job: Control plane -> tracking database.
    task_metrics: A task's records shaped for the app's metrics panel.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from logic.controllers.cloud.interfaces import ControlPlane

_PREFIX = "GUI_DAY_LOG_START:"
_PROGRESS_KPIS = ("kg", "km", "kg/km", "overflows", "kg_lost", "profit", "ncol", "time", "day_wall_time")


class ResourceSampler:
    """Samples a process and its children with psutil.

    Args:
        pid: Root process id (the simulator).
        workdir: Directory whose size is reported.
        clock: Time source.
    """

    def __init__(self, pid: int, workdir: Path, clock: Callable[[], float] = time.time) -> None:
        import psutil  # optional: TaskTelemetry skips resource sampling without it

        self._psutil = psutil
        self.workdir = Path(workdir)
        self.clock = clock
        self.root: Optional[Any]
        try:
            self.root = psutil.Process(pid)
        except psutil.NoSuchProcess:
            self.root = None
        self._net0 = psutil.net_io_counters()

    def sample(self) -> Optional[Dict[str, Any]]:
        """One resource record, or None once the process is gone."""
        psutil = self._psutil
        if self.root is None:
            return None
        try:
            procs = [self.root, *self.root.children(recursive=True)]
        except psutil.NoSuchProcess:
            return None
        cpu = rss = threads = 0.0
        for proc in procs:
            try:
                with proc.oneshot():
                    cpu += proc.cpu_percent(interval=None)
                    rss += proc.memory_info().rss
                    threads += proc.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        net = psutil.net_io_counters()
        disk = sum(p.stat().st_size for p in self.workdir.rglob("*") if p.is_file()) if self.workdir.is_dir() else 0
        return {
            "at": self.clock(),
            "kind": "resource",
            "cpu_percent": round(cpu, 1),
            "memory_rss_mb": round(rss / 2**20, 1),
            "threads": int(threads),
            "processes": len(procs),
            "disk_mb": round(disk / 2**20, 1),
            "net_rx_mb": round((net.bytes_recv - self._net0.bytes_recv) / 2**20, 3),
            "net_tx_mb": round((net.bytes_sent - self._net0.bytes_sent) / 2**20, 3),
        }


class ProgressReader:
    """Reads new day records from the realtime logs under an output directory.

    Args:
        outputs: The simulator's output directory.
        days: The cell's horizon (``sim.graph.n_days``), if known.
        clock: Time source.
    """

    def __init__(self, outputs: Path, days: Optional[int] = None, clock: Callable[[], float] = time.time) -> None:
        self.outputs = Path(outputs)
        self.days = days
        self.clock = clock
        self._offsets: Dict[Path, int] = {}
        self.latest: Dict[str, Any] = {}

    def read(self) -> List[Dict[str, Any]]:
        """Progress records for day lines written since the last read."""
        records = []
        for path in sorted(self.outputs.rglob("log_realtime_*.jsonl")) if self.outputs.is_dir() else []:
            offset = self._offsets.get(path, 0)
            with open(path, "rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
            complete = chunk[: chunk.rfind(b"\n") + 1]  # a line still being written is read next time
            self._offsets[path] = offset + len(complete)
            for line in complete.decode(errors="replace").splitlines():
                record = self._parse(line)
                if record is not None:
                    records.append(record)
        if records:
            last = records[-1]
            self.latest = {"day": last["day"], "days": self.days, "policy": last["policy"]}
        return records

    def _parse(self, line: str) -> Optional[Dict[str, Any]]:
        if not line.startswith(_PREFIX):
            return None
        head, sep, payload = line[len(_PREFIX) :].partition(",{")
        parts = head.rsplit(",", 2)
        if not sep or len(parts) != 3:
            return None
        policy, sample, day = parts
        try:
            data = json.loads("{" + payload)
        except json.JSONDecodeError:
            return None
        kpis = {key: data[key] for key in _PROGRESS_KPIS if isinstance(data.get(key), (int, float))}
        return {
            "at": self.clock(),
            "kind": "progress",
            "policy": policy,
            "sample": int(sample) if sample.isdigit() else sample,
            "day": int(day) if day.isdigit() else day,
            "days": self.days,
            "solver_status": data.get("solver_status"),
            "kpis": kpis,
        }


class TelemetryBuffer:
    """Batches records into ``put_metrics``, at most once per ``flush_seconds``.

    Args:
        cp: Control plane.
        task_id: Task the records belong to.
        flush_seconds: Minimum seconds between writes.
        max_batch: Records per write; extra records wait for the next flush.
        clock: Time source.
    """

    def __init__(
        self,
        cp: ControlPlane,
        task_id: str,
        flush_seconds: float = 10.0,
        max_batch: int = 200,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.cp, self.task_id = cp, task_id
        self.flush_seconds, self.max_batch, self.clock = flush_seconds, max_batch, clock
        self.pending: List[Dict[str, Any]] = []
        self._last_flush = -float("inf")
        self.written = 0

    def add(self, records: List[Dict[str, Any]]) -> None:
        self.pending.extend(r for r in records if r)

    def flush(self, force: bool = False) -> int:
        """Write pending records if the interval has passed (or ``force``); returns how many."""
        if not self.pending or (not force and self.clock() - self._last_flush < self.flush_seconds):
            return 0
        written = 0
        while self.pending and (force or written < self.max_batch):
            batch, self.pending = self.pending[: self.max_batch], self.pending[self.max_batch :]
            self.cp.put_metrics(self.task_id, batch)
            written += len(batch)
            if not force:
                break
        self._last_flush = self.clock()
        self.written += written
        return written


def import_job(
    cp: ControlPlane, job_id: str, store: Any, experiment: str = "cloud", results_dir: Optional[Path] = None
) -> List[str]:
    """Copy a job's tasks, metrics and events into a tracking database; returns the run ids.

    Re-importing a job replaces its runs' metrics and events, so it is safe to repeat.

    Args:
        cp: Control plane.
        job_id: Job to import.
        store: A :class:`logic.src.tracking.core.store.TrackingStore`.
        experiment: Tracking experiment name.
        results_dir: Where ``cloud download`` puts the job's outputs; recorded as each
            run's artifact directory, linking the tracking run to the local results.
    """
    job = cp.get_job(job_id)
    experiment_id = store.get_or_create_experiment(experiment, description="Cloud runner jobs")
    events = cp.list_events(job_id)
    run_ids = []
    for task in cp.list_tasks(job_id):
        run_id = f"cloud-{task.task_id}"
        if store.get_run(run_id) is None:
            artifact_dir = str(results_dir) if results_dir is not None else ""
            store.create_run(run_id, experiment_id, f"{job.spec.name}/{task.cell.cell_id}", "cloud_task", artifact_dir)
        store.clear_run_data(run_id)
        store.set_tags(
            run_id,
            {
                "job_id": job_id,
                "task_id": task.task_id,
                "cell_id": task.cell.cell_id,
                "provider": str(getattr(job.spec.provider, "value", job.spec.provider)),
                "compute": str(getattr(job.spec.compute, "value", job.spec.compute)),
                "attempt": str(task.attempt),
                "state": task.state.value,
            },
        )
        store.log_params(run_id, {"argv": " ".join(task.cell.argv), "max_retries": job.spec.resources.max_retries})
        metrics = []
        for index, record in enumerate(cp.get_metrics(task.task_id)):
            kind = record.get("kind")
            if kind == "resource":
                metrics += [(f"resource/{k}", float(v), index) for k, v in record.items() if k not in ("at", "kind")]
            elif kind == "progress" and isinstance(record.get("day"), int):
                metrics += [(f"progress/{k}", float(v), record["day"]) for k, v in record.get("kpis", {}).items()]
            elif kind == "cost":
                metrics += [
                    (f"cost/{k}", float(v), index)
                    for k, v in record.items()
                    if isinstance(v, (int, float)) and k != "at"
                ]
        if metrics:
            store.log_metrics_batch(run_id, metrics)
        for event in events:
            if event.task_id in (None, task.task_id):
                store.log_run_event(run_id, event.kind, event.data, event.at)
        if task.state.value in ("succeeded", "failed", "cancelled"):
            store.finish_run(
                run_id, status="completed" if task.state.value == "succeeded" else task.state.value, error=task.message
            )
        run_ids.append(run_id)
    return run_ids


class TaskTelemetry:
    """Resource sampling, progress reading and batched writes for one task attempt.

    Every call is best effort: an instrumentation error is recorded once in the
    task message stream and never stops the simulation.

    Args:
        cp: Control plane.
        task_id: The task.
        outputs: The simulator's output directory.
        workdir: The task's work directory (its size is reported).
        days: The cell's horizon, if known.
        sample_seconds: Seconds between resource samples and progress reads.
        flush_seconds: Minimum seconds between control-plane writes.
        clock: Time source.
    """

    def __init__(
        self,
        cp: ControlPlane,
        task_id: str,
        outputs: Path,
        workdir: Path,
        days: Optional[int] = None,
        sample_seconds: float = 5.0,
        flush_seconds: float = 10.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.workdir, self.sample_seconds, self.clock = Path(workdir), sample_seconds, clock
        self.progress = ProgressReader(outputs, days=days, clock=clock)
        self.buffer = TelemetryBuffer(cp, task_id, flush_seconds=flush_seconds, clock=clock)
        self.sampler: Optional[ResourceSampler] = None
        self.errors: List[str] = []
        self._last_sample = -float("inf")

    def attach(self, pid: int) -> None:
        """Start sampling the simulator process ``pid`` (no-op without psutil)."""
        try:
            self.sampler = ResourceSampler(pid, self.workdir, clock=self.clock)
        except ImportError:
            self.sampler = None
        except Exception as exc:  # noqa: BLE001  (instrumentation must not stop the task)
            self._error(exc)

    def tick(self, now: Optional[float] = None) -> None:
        """Sample and read progress when ``sample_seconds`` have passed, then flush if due."""
        now = self.clock() if now is None else now
        if now - self._last_sample < self.sample_seconds:
            return
        self._last_sample = now
        try:
            if self.sampler is not None:
                record = self.sampler.sample()
                if record is not None:
                    self.buffer.add([record])
            self.buffer.add(self.progress.read())
            self.buffer.flush()
        except Exception as exc:  # noqa: BLE001
            self._error(exc)

    def close(self) -> None:
        """Read the last progress lines and write everything still pending."""
        try:
            self.buffer.add(self.progress.read())
            self.buffer.flush(force=True)
        except Exception as exc:  # noqa: BLE001
            self._error(exc)

    def _error(self, exc: Exception) -> None:
        if len(self.errors) < 5:
            self.errors.append(f"{type(exc).__name__}: {exc}")


def days_from_argv(argv: List[str]) -> Optional[int]:
    """The horizon of a ``test_sim`` command line (``sim.graph.n_days=N``, ``sim.days=N``, or ``days=N``), if given."""
    for arg in argv:
        key, sep, value = arg.partition("=")
        clean_key = key.lstrip("+")
        clean_val = value.strip("'\" ")
        if sep and clean_key in ("sim.graph.n_days", "sim.days", "days") and clean_val.isdigit():
            return int(clean_val)
    return None


def task_metrics(records: List[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
    """A task's records in the shape of the app's ``TaskMetricsData`` (resource points, peaks, progress)."""
    points = [
        {
            "timestamp_ms": int(r["at"] * 1000),
            "cpu_percent": float(r.get("cpu_percent", 0.0)),
            "memory_rss_mb": float(r.get("memory_rss_mb", 0.0)),
        }
        for r in records
        if r.get("kind") == "resource"
    ]
    progress = [r for r in records if r.get("kind") == "progress"]
    last = progress[-1] if progress else None
    return {
        "task_id": task_id,
        "points": points,
        "peak_cpu": max((p["cpu_percent"] for p in points), default=None),
        "peak_memory_mb": max((p["memory_rss_mb"] for p in points), default=None),
        "progress": None
        if last is None
        else {"day": last["day"], "days": last.get("days"), "policy": last["policy"], "kpis": last.get("kpis", {})},
        "days": [
            {"policy": r["policy"], "day": r["day"], "solver_status": r.get("solver_status"), **r.get("kpis", {})}
            for r in progress
        ],
    }
