"""Naming, labels, error mapping and logs shared by the GCP backends (CR-05).

Attributes:
    resource_name: ``wsr-<task>-a<attempt>``, valid for Batch jobs, Cloud Run jobs and VMs.
    labels: Labels that tie a resource to its job, task and attempt (``wsr_job_id`` is also
        what billing exports group by, CR-10).
    raise_capacity: Re-raise quota and stock-out errors as :class:`CapacityError`.
    runtime_log: The attempt log the task runtime uploads to object storage.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from logic.controllers.cloud.interfaces import CapacityError, LaunchSpec

_CAPACITY = ("ResourceExhausted", "TooManyRequests", "ZONE_RESOURCE_POOL_EXHAUSTED", "QUOTA_EXCEEDED", "STOCKOUT")


def resource_name(launch: LaunchSpec) -> str:
    """A lowercase name of at most 63 characters starting with a letter."""
    raw = f"wsr-{launch.task.task_id}-a{launch.task.attempt}".lower()
    return re.sub(r"[^a-z0-9-]", "-", raw)[:63].rstrip("-")


def _label(value: Any) -> str:
    return re.sub(r"[^a-z0-9_-]", "_", str(value).lower())[:63]


def labels(launch: LaunchSpec) -> Dict[str, str]:
    return {
        "wsr_job_id": _label(launch.task.job_id),
        "wsr_task_id": _label(launch.task.task_id),
        "wsr_attempt": _label(launch.task.attempt),
    }


def raise_capacity(exc: Exception) -> None:
    """Raise ``CapacityError`` for quota or stock-out failures; otherwise return (the caller re-raises)."""
    text = f"{type(exc).__name__} {exc}"
    if getattr(exc, "code", None) == 429 or any(marker in text for marker in _CAPACITY):
        raise CapacityError(str(exc)) from exc


def runtime_log(store: Optional[Any], resource_labels: Dict[str, str], tail: Optional[int]) -> str:
    """The runtime's uploaded ``work/<job>/<task>/logs/attempt-<n>.log``, or ``""``."""
    if store is None or not resource_labels:
        return ""
    key = (
        f"work/{resource_labels.get('wsr_job_id')}/{resource_labels.get('wsr_task_id')}"
        f"/logs/attempt-{resource_labels.get('wsr_attempt')}.log"
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "log.txt"
        try:
            store.get_file(key, target)
        except KeyError:
            return ""
        lines = target.read_text(errors="replace").splitlines()
    return "\n".join(lines if tail is None else lines[-tail:])
