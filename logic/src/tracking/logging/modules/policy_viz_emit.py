"""Policy telemetry bridge for WSmart-Route Studio (§A.3).

Serialises :class:`~logic.src.tracking.viz_mixin.PolicyVizMixin` ring-buffer
data into ``POLICY_VIZ_START:`` log lines recognised by the Tauri Studio app.

Example:
    >>> maybe_emit_policy_viz(policy, "ALNS + Ftsp", 0, 1, "run.jsonl")
"""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, Dict, List, Optional

import logic.src.constants as udef
from logic.src.tracking.logging.modules.policy_telemetry_db import persist_policy_viz_snapshot
from logic.src.utils.infrastructure.setup_sims import deep_sanitize

POLICY_VIZ_MARKER = "POLICY_VIZ_START:"
STREAM_INTERVAL_SEC = 0.5  # 2 Hz refresh for §A.3 Option B
# The Studio sets this when it launches a run. Without it nobody reads the live stream, and each
# snapshot repeats the whole ring buffer, so a console run of ALNS printed tens of MB of it.
STUDIO_ENV = "WSR_STUDIO"


def studio_attached() -> bool:
    """Whether WSmart-Route Studio launched this process and reads its live telemetry."""
    return os.environ.get(STUDIO_ENV, "") not in ("", "0")


def detect_policy_viz_type(viz_data: Dict[str, List[Any]]) -> str:
    """Infer algorithm family from telemetry key signatures.

    Consumed by the WSmart-Route Studio policy telemetry panels (app/).

    Args:
        viz_data: Mapping returned by ``get_viz_data()``.

    Returns:
        One of ``alns``, ``hgs``, ``aco``, ``ils``, ``selector``, or ``generic``.
    """
    if "d_idx" in viz_data:
        return "alns"
    if "generation" in viz_data:
        return "hgs"
    if "tau_mean" in viz_data:
        return "aco"
    if "perturb_mode" in viz_data:
        return "ils"
    if "n_selected" in viz_data:
        return "selector"
    return "generic"


def _collect_viz_sources(source: Any) -> List[Any]:
    """Gather objects that may expose ``get_viz_data()``."""
    candidates: List[Any] = []
    seen: set[int] = set()

    def add(obj: Any) -> None:
        if obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        candidates.append(obj)

    add(source)
    for attr in ("engine", "solver", "policy", "_solver", "_engine", "improver"):
        add(getattr(source, attr, None))
    return candidates


class PolicyVizStreamSession:
    """Emit growing ring-buffer snapshots at 2 Hz during solver execution (§A.3 Option B).

    Wraps routing / improvement so the Studio receives live telemetry via
    ``process:stdout`` and ``sim:policy_viz_update`` without waiting for solver completion.
    Outside the Studio (see :func:`studio_attached`) only the final snapshot is written.
    """

    def __init__(
        self,
        source: Any,
        policy: str,
        sample_idx: int,
        day: int,
        log_path: Optional[str],
        lock: Optional[threading.Lock] = None,
        interval_sec: float = STREAM_INTERVAL_SEC,
    ) -> None:
        self._source = source
        self._policy = policy
        self._sample_idx = sample_idx
        self._day = day
        self._log_path = log_path
        self._lock = lock
        self._interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "PolicyVizStreamSession":
        if not studio_attached():
            return self
        self._thread = threading.Thread(
            target=self._emit_loop,
            name="policy-viz-stream",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        maybe_emit_policy_viz(
            self._source,
            self._policy,
            self._sample_idx,
            self._day,
            self._log_path,
            self._lock,
        )

    def _emit_loop(self) -> None:
        while not self._stop.wait(self._interval_sec):
            maybe_emit_policy_viz(
                self._source,
                self._policy,
                self._sample_idx,
                self._day,
                self._log_path,
                self._lock,
            )


def maybe_emit_policy_viz(
    source: Any,
    policy: str,
    sample_idx: int,
    day: int,
    log_path: Optional[str],
    lock: Optional[threading.Lock] = None,
) -> bool:
    """Emit policy telemetry when ``source`` (or a nested engine) recorded data.

    Args:
        source: Policy, adapter, or route improver instance.
        policy: Display policy name written to the log line.
        sample_idx: Simulation sample index.
        day: Simulation day (1-indexed).
        log_path: Optional JSONL path for append; stdout is written when the Studio is attached.
        lock: Optional thread lock for safe file appending.

    Returns:
        True when telemetry was emitted.
    """
    for obj in _collect_viz_sources(source):
        getter = getattr(obj, "get_viz_data", None)
        if getter is None:
            continue
        try:
            viz_data = getter()
        except Exception:
            continue
        if isinstance(viz_data, dict) and viz_data:
            send_policy_viz_to_gui(viz_data, policy, sample_idx, day, log_path, lock)
            return True
    return False


def send_policy_viz_to_gui(
    viz_data: Dict[str, List[Any]],
    policy: str,
    sample_idx: int,
    day: int,
    log_path: Optional[str],
    lock: Optional[threading.Lock] = None,
) -> None:
    """Write a ``POLICY_VIZ_START:`` line to the simulation log file (and stdout for the Studio).

    Args:
        viz_data: Telemetry dict from ``PolicyVizMixin.get_viz_data()``.
        policy: Policy display name.
        sample_idx: Sample index.
        day: Simulation day.
        log_path: Optional JSONL path.
        lock: Optional thread lock.
    """
    if not viz_data:
        return

    payload = deep_sanitize(viz_data)
    policy_type = detect_policy_viz_type(payload)
    json_payload = json.dumps(payload, separators=(",", ":"))
    log_msg = f"{POLICY_VIZ_MARKER}{policy},{sample_idx},{day},{policy_type},{json_payload}"

    if studio_attached():
        print(log_msg, flush=True)

    try:
        persist_policy_viz_snapshot(
            payload, policy, sample_idx, day, policy_type, log_path
        )
    except Exception as exc:
        print(f"Warning: Failed to persist policy telemetry: {exc}", file=sys.stderr)

    if not log_path:
        return

    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return
    try:
        with open(log_path, "a") as f:
            f.write(log_msg + "\n")
            f.flush()
    except OSError as exc:
        print(f"Warning: Failed to write policy viz to log file: {exc}", file=sys.stderr)
    finally:
        if lock is not None:
            lock.release()
