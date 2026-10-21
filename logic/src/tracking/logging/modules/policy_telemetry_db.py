"""SQLite persistence for policy telemetry cross-run trending (§A.3 Option C).

Ring-buffer snapshots from :class:`~logic.src.tracking.viz_mixin.PolicyVizMixin` are
stored in ``assets/telemetry.db`` so the Studio can compare solver trajectories across
simulation runs without re-parsing JSONL logs.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from logic.src.constants import ROOT_DIR

_DB_LOCK = threading.Lock()
_SCHEMA_VERSION = 1

base_uri = "test_tracking" if os.environ.get("TEST_MODE") == "true" else "assets"
TELEMETRY_DB_PATH: str = str(Path(ROOT_DIR).joinpath(base_uri, "telemetry.db").resolve())


def _conn(timeout: float = 5.0) -> sqlite3.Connection:
    Path(TELEMETRY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TELEMETRY_DB_PATH, timeout=timeout)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS simulation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            log_path TEXT NOT NULL UNIQUE,
            run_label TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS policy_viz_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL REFERENCES simulation_runs(id) ON DELETE CASCADE,
            policy TEXT NOT NULL,
            sample_idx INTEGER NOT NULL,
            day INTEGER NOT NULL,
            policy_type TEXT NOT NULL,
            step_count INTEGER NOT NULL,
            final_metric REAL,
            metric_name TEXT,
            data_json TEXT NOT NULL,
            emitted_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_policy_viz_run
            ON policy_viz_snapshots(run_id);
        CREATE INDEX IF NOT EXISTS idx_policy_viz_policy_type
            ON policy_viz_snapshots(policy_type);
        CREATE INDEX IF NOT EXISTS idx_policy_viz_policy_day
            ON policy_viz_snapshots(policy, day);
        """
    )
    conn.execute(
        "INSERT INTO schema_meta (key, value) VALUES ('version', ?) "
        "ON CONFLICT(key) DO NOTHING",
        (str(_SCHEMA_VERSION),),
    )
    conn.commit()


def _series_tail(values: Any) -> Optional[float]:
    if not isinstance(values, list) or not values:
        return None
    last = values[-1]
    if isinstance(last, bool):
        return None
    try:
        return float(last)
    except (TypeError, ValueError):
        return None


def extract_final_metric(
    policy_type: str, viz_data: Dict[str, List[Any]]
) -> Tuple[Optional[float], Optional[str]]:
    """Pick a comparable scalar from the terminal ring-buffer values."""
    metric_map = {
        "alns": "best_cost",
        "hgs": "best_cost",
        "aco": "global_best_cost",
        "ils": "best_cost",
        "selector": "n_selected",
    }
    key = metric_map.get(policy_type)
    if key and key in viz_data:
        return _series_tail(viz_data[key]), key
    for fallback in ("best_cost", "global_best_cost", "current_cost", "mean_cost"):
        if fallback in viz_data:
            return _series_tail(viz_data[fallback]), fallback
    for name, values in viz_data.items():
        if name in ("iteration", "generation", "restart", "op_name", "d_idx", "r_idx"):
            continue
        val = _series_tail(values)
        if val is not None:
            return val, name
    return None, None


def _step_count(viz_data: Dict[str, List[Any]]) -> int:
    lengths = [len(v) for v in viz_data.values() if isinstance(v, list)]
    return max(lengths) if lengths else 0


def _run_label_from_path(log_path: Optional[str]) -> Optional[str]:
    if not log_path:
        return None
    stem = Path(log_path).stem
    return stem or None


def _upsert_run(conn: sqlite3.Connection, log_path: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    label = _run_label_from_path(log_path)
    conn.execute(
        """
        INSERT INTO simulation_runs (log_path, run_label, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(log_path) DO UPDATE SET run_label = excluded.run_label
        """,
        (log_path, label, now),
    )
    row = conn.execute(
        "SELECT id FROM simulation_runs WHERE log_path = ?", (log_path,)
    ).fetchone()
    assert row is not None
    return int(row["id"])


def persist_policy_viz_snapshot(
    viz_data: Dict[str, List[Any]],
    policy: str,
    sample_idx: int,
    day: int,
    policy_type: str,
    log_path: Optional[str],
) -> bool:
    """Insert or replace the latest snapshot for a run × policy × sample × day."""
    if not viz_data or not log_path:
        return False

    final_metric, metric_name = extract_final_metric(policy_type, viz_data)
    step_count = _step_count(viz_data)
    payload = json.dumps(viz_data, separators=(",", ":"))
    now = datetime.now(timezone.utc).isoformat()

    with _DB_LOCK:
        conn = _conn()
        try:
            _ensure_schema(conn)
            run_id = _upsert_run(conn, log_path)
            conn.execute(
                """
                DELETE FROM policy_viz_snapshots
                WHERE run_id = ? AND policy = ? AND sample_idx = ? AND day = ?
                """,
                (run_id, policy, sample_idx, day),
            )
            conn.execute(
                """
                INSERT INTO policy_viz_snapshots (
                    run_id, policy, sample_idx, day, policy_type,
                    step_count, final_metric, metric_name, data_json, emitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    policy,
                    sample_idx,
                    day,
                    policy_type,
                    step_count,
                    final_metric,
                    metric_name,
                    payload,
                    now,
                ),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def _x_axis_from_viz(viz_data: Dict[str, List[Any]]) -> List[int]:
    for key in ("iteration", "generation", "restart"):
        values = viz_data.get(key)
        if isinstance(values, list) and values:
            out: List[int] = []
            for idx, raw in enumerate(values):
                if isinstance(raw, bool):
                    out.append(idx)
                else:
                    try:
                        out.append(int(raw))
                    except (TypeError, ValueError):
                        out.append(idx)
            return out
    lengths = [len(v) for v in viz_data.values() if isinstance(v, list)]
    span = max(lengths) if lengths else 0
    return list(range(span))


def _metric_series(
    policy_type: str, viz_data: Dict[str, List[Any]]
) -> Tuple[Optional[str], List[float]]:
    _, metric_name = extract_final_metric(policy_type, viz_data)
    if not metric_name or metric_name not in viz_data:
        return None, []
    values: List[float] = []
    for raw in viz_data[metric_name]:
        if isinstance(raw, bool):
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return metric_name, values


def query_policy_trajectory_series(
    policy: Optional[str] = None,
    policy_type: Optional[str] = None,
    run_label: Optional[str] = None,
    limit: int = 12,
) -> Dict[str, Any]:
    """Return improvement curves extracted from persisted ring-buffer JSON."""
    with _DB_LOCK:
        if not os.path.exists(TELEMETRY_DB_PATH):
            return {"db_path": TELEMETRY_DB_PATH, "series": []}

        conn = _conn()
        try:
            _ensure_schema(conn)
            sql = """
                SELECT
                    s.id,
                    r.run_label,
                    r.log_path,
                    s.policy,
                    s.day,
                    s.policy_type,
                    s.metric_name,
                    s.data_json
                FROM policy_viz_snapshots s
                JOIN simulation_runs r ON r.id = s.run_id
                WHERE 1=1
            """
            params: List[Any] = []
            if policy:
                sql += " AND s.policy = ?"
                params.append(policy)
            if policy_type:
                sql += " AND s.policy_type = ?"
                params.append(policy_type)
            if run_label:
                sql += " AND r.run_label = ?"
                params.append(run_label)
            sql += " ORDER BY s.emitted_at DESC LIMIT ?"
            params.append(max(1, limit))

            series: List[Dict[str, Any]] = []
            for row in conn.execute(sql, params).fetchall():
                try:
                    viz_data = json.loads(row["data_json"])
                except json.JSONDecodeError:
                    continue
                if not isinstance(viz_data, dict):
                    continue

                metric_name, y_values = _metric_series(row["policy_type"], viz_data)
                if not metric_name or len(y_values) < 2:
                    continue

                x_values = _x_axis_from_viz(viz_data)
                if len(x_values) != len(y_values):
                    x_values = list(range(len(y_values)))

                run_label = row["run_label"] or Path(row["log_path"]).stem
                series.append(
                    {
                        "id": int(row["id"]),
                        "label": f"{run_label} · d{int(row['day'])}",
                        "run_label": run_label,
                        "policy": row["policy"],
                        "day": int(row["day"]),
                        "policy_type": row["policy_type"],
                        "metric_name": metric_name,
                        "x": x_values,
                        "y": y_values,
                    }
                )
            return {"db_path": TELEMETRY_DB_PATH, "series": series}
        finally:
            conn.close()


def query_policy_telemetry_trends(
    policy_type: Optional[str] = None,
    run_label: Optional[str] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    """Return cross-run snapshot rows for the Studio trends panel."""
    with _DB_LOCK:
        if not os.path.exists(TELEMETRY_DB_PATH):
            return {"db_path": TELEMETRY_DB_PATH, "rows": [], "policy_types": []}

        conn = _conn()
        try:
            _ensure_schema(conn)
            types = [
                r["policy_type"]
                for r in conn.execute(
                    """
                    SELECT DISTINCT policy_type FROM policy_viz_snapshots
                    ORDER BY policy_type
                    """
                ).fetchall()
            ]

            sql = """
                SELECT
                    s.id,
                    r.log_path,
                    r.run_label,
                    r.created_at AS run_created_at,
                    s.policy,
                    s.sample_idx,
                    s.day,
                    s.policy_type,
                    s.step_count,
                    s.final_metric,
                    s.metric_name,
                    s.emitted_at
                FROM policy_viz_snapshots s
                JOIN simulation_runs r ON r.id = s.run_id
            """
            params: List[Any] = []
            clauses: List[str] = []
            if policy_type:
                clauses.append("s.policy_type = ?")
                params.append(policy_type)
            if run_label:
                clauses.append("r.run_label = ?")
                params.append(run_label)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY s.emitted_at DESC LIMIT ?"
            params.append(max(1, limit))

            rows = [
                {
                    "id": int(r["id"]),
                    "log_path": r["log_path"],
                    "run_label": r["run_label"],
                    "run_created_at": r["run_created_at"],
                    "policy": r["policy"],
                    "sample_idx": int(r["sample_idx"]),
                    "day": int(r["day"]),
                    "policy_type": r["policy_type"],
                    "step_count": int(r["step_count"]),
                    "final_metric": r["final_metric"],
                    "metric_name": r["metric_name"],
                    "emitted_at": r["emitted_at"],
                }
                for r in conn.execute(sql, params).fetchall()
            ]
            return {
                "db_path": TELEMETRY_DB_PATH,
                "rows": rows,
                "policy_types": types,
            }
        finally:
            conn.close()
