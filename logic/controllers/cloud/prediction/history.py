"""Training data and accuracy tracking from completed cloud jobs (CR-11).

Attributes:
    cloud_records: Completed single-attempt tasks as training rows, with their peak memory.
    fitted_predictor: The bundled table plus the control plane's completed tasks.
    record_predictions: Store a job's predictions as a ``prediction`` event.
    prediction_accuracy: Predicted against actual wall time for a job's finished tasks.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple

from logic.controllers.cloud.interfaces import ControlPlane
from logic.controllers.cloud.model import Event, TaskState
from logic.controllers.cloud.prediction.features import cell_features
from logic.controllers.cloud.prediction.model import (
    MIN_MACHINE_SAMPLES,
    Prediction,
    RuntimePredictor,
    load_table,
)


def _peak_mb(cp: ControlPlane, task_id: str) -> Optional[float]:
    peaks = [
        float(r["memory_rss_mb"])
        for r in cp.get_metrics(task_id)
        if r.get("kind") == "resource" and r.get("memory_rss_mb")
    ]
    return max(peaks) if peaks else None


def cloud_records(cp: ControlPlane) -> List[Dict[str, Any]]:
    """Training rows of every succeeded task that finished in its first attempt.

    A retried task's ``started_at`` is its last attempt, so its total time is unknown; it is skipped.
    Rows carry ``source="cloud"`` and ``peak_memory_mb`` (CR-12 telemetry) when recorded.
    """
    rows = []
    for job in cp.list_jobs():
        machine = job.spec.resources.machine_type
        for task in cp.list_tasks(job.job_id):
            if (
                task.state != TaskState.SUCCEEDED
                or task.attempt != 1
                or task.started_at is None
                or task.finished_at is None
            ):
                continue
            wall = task.finished_at - task.started_at
            if wall <= 0:
                continue
            row = cell_features(task.cell, machine)
            row.update(
                wall_seconds=wall, source="cloud", cell_id=task.cell.cell_id, peak_memory_mb=_peak_mb(cp, task.task_id)
            )
            rows.append(row)
    return rows


def _machine_factors(base: RuntimePredictor, rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Median observed/predicted laptop time per machine type with enough samples."""
    by_machine: Dict[str, List[float]] = {}
    for row, q in zip(rows, base.predict_features(rows), strict=True):
        if q is not None and row.get("machine_type") not in (None, "unknown"):
            by_machine.setdefault(row["machine_type"], []).append(row["wall_seconds"] / q[1])
    return {m: round(statistics.median(r), 4) for m, r in by_machine.items() if len(r) >= MIN_MACHINE_SAMPLES}


def fitted_predictor(cp: Optional[ControlPlane] = None) -> RuntimePredictor:
    """The bundled table's model, with machine factors and memory peaks from ``cp``'s completed tasks.

    Cloud rows are not mixed into the laptop-time regression: they set each machine's
    speed factor against it, so a few fast or slow machines cannot distort the laptop model.
    """
    base = RuntimePredictor(load_table())
    if cp is None:
        return base
    rows = cloud_records(cp)
    peaks: Dict[Tuple[str, Optional[int]], List[float]] = {}
    for row in rows:
        if row.get("peak_memory_mb"):
            peaks.setdefault((row["policy"], row["size"]), []).append(row["peak_memory_mb"])
    base.machine_factors = _machine_factors(base, rows)
    base.memory_peaks = peaks
    return base


def record_predictions(cp: ControlPlane, job_id: str, predictions: Sequence[Prediction], at: float) -> None:
    """Store a job's predictions so their accuracy can be checked after the job."""
    cp.append_event(
        Event(job_id=job_id, kind="prediction", data={"cells": {p.cell_id: p.to_dict() for p in predictions}}, at=at)
    )


def prediction_accuracy(cp: ControlPlane, job_id: str) -> Dict[str, Any]:
    """Predicted against actual wall time of a job's succeeded first-attempt tasks.

    Returns per-task rows (``predicted_p50``, ``actual``, ``ratio``, ``inside`` the 10-90 % interval)
    and the summary ``median_ape`` and ``coverage``; empty when the job has no prediction event.
    """
    events = [e for e in cp.list_events(job_id) if e.kind == "prediction"]
    if not events:
        return {"job_id": job_id, "tasks": [], "median_ape": None, "coverage": None}
    predicted = events[-1].data.get("cells", {})
    rows = []
    for task in cp.list_tasks(job_id):
        p = predicted.get(task.cell.cell_id)
        if (
            p is None
            or p.get("p50") is None
            or task.state != TaskState.SUCCEEDED
            or task.attempt != 1
            or task.started_at is None
            or task.finished_at is None
        ):
            continue
        actual = task.finished_at - task.started_at
        rows.append(
            {
                "task_id": task.task_id,
                "cell_id": task.cell.cell_id,
                "predicted_p50": p["p50"],
                "predicted_p10": p["p10"],
                "predicted_p90": p["p90"],
                "actual": round(actual, 1),
                "ratio": round(actual / p["p50"], 3) if p["p50"] else None,
                "inside": p["p10"] <= actual <= p["p90"],
                "source": p.get("source"),
            }
        )
    ape = [abs(r["actual"] - r["predicted_p50"]) / r["actual"] for r in rows if r["actual"] > 0]
    return {
        "job_id": job_id,
        "tasks": rows,
        "median_ape": round(statistics.median(ape), 4) if ape else None,
        "coverage": round(sum(r["inside"] for r in rows) / len(rows), 3) if rows else None,
    }
