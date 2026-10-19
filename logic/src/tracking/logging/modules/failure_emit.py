"""Simulation failure analysis bridge for WSmart-Route Studio (§A.6).

Serialises :class:`~logic.src.tracking.logging.modules.failure_analyzer.FailureAnalyzer`
summaries into ``SIM_FAILURE_START:`` log lines recognised by the Tauri Studio app.

Example:
    >>> emit_sim_failure_summary({"has_failure": True}, "greedy", 0, 1)
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, Dict, Optional

import logic.src.constants as udef
from logic.src.utils.infrastructure.setup_sims import deep_sanitize

SIM_FAILURE_MARKER = "SIM_FAILURE_START:"


def emit_sim_failure_summary(
    summary: Dict[str, Any],
    policy: str,
    sample_idx: int,
    day: int,
    log_path: Optional[str] = None,
    lock: Optional[threading.Lock] = None,
) -> None:
    """Write a ``SIM_FAILURE_START:`` line to stdout and the simulation log file.

    Args:
        summary: Structured failure summary from ``FailureAnalyzer``.
        policy: Policy display name.
        sample_idx: Simulation sample index.
        day: Simulation day (1-indexed).
        log_path: Optional JSONL path for append.
        lock: Optional thread lock for safe file appending.
    """
    if not summary.get("has_failure"):
        return

    payload = deep_sanitize(summary)
    json_payload = json.dumps(payload, separators=(",", ":"))
    log_msg = f"{SIM_FAILURE_MARKER}{policy},{sample_idx},{day},{json_payload}"

    print(log_msg, flush=True)

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
        print(f"Warning: Failed to write failure analysis to log file: {exc}", file=sys.stderr)
    finally:
        if lock is not None:
            lock.release()
