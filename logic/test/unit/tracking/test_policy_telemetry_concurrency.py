"""Independent simulation processes must initialize telemetry without losing rows."""

import multiprocessing
import sqlite3

from logic.src.tracking.logging.modules import policy_telemetry_db as db


def _write_snapshot(path, barrier, results, policy):
    # Each spawned process imports its own module and owns a separate _DB_LOCK.
    db.TELEMETRY_DB_PATH = path
    connect = db._conn

    def synchronized_connection():
        connection = connect(timeout=15.0)
        synchronized = False

        def trace(statement):
            nonlocal synchronized
            # SQLite begins a deferred transaction immediately before the first
            # INSERT. Both old SELECTs have completed by this point, forcing the
            # initialization race without faking any query results or locks.
            if statement.strip().upper().startswith("BEGIN") and not synchronized:
                synchronized = True
                barrier.wait(timeout=15.0)

        connection.set_trace_callback(trace)
        return connection

    db._conn = synchronized_connection
    try:
        saved = db.persist_policy_viz_snapshot(
            {"iteration": [0], "best_cost": [10.0]}, policy, 0, 1, "alns", "/tmp/concurrent-run.jsonl"
        )
        results.put((policy, saved, None))
    except Exception as error:
        results.put((policy, False, repr(error)))


def test_fresh_database_initialization_is_atomic_across_processes(tmp_path):
    path = str(tmp_path / "telemetry.db")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(target=_write_snapshot, args=(path, barrier, results, policy)) for policy in ("A", "B")
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30.0)
            assert process.exitcode == 0
        outcomes = [results.get(timeout=5.0) for _ in processes]
        assert sorted(outcomes) == [("A", True, None), ("B", True, None)]
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT value FROM schema_meta WHERE key='version'").fetchall() == [("1",)]
            assert connection.execute("SELECT policy FROM policy_viz_snapshots ORDER BY policy").fetchall() == [
                ("A",),
                ("B",),
            ]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        results.close()


def test_schema_initialization_preserves_existing_version(tmp_path):
    with sqlite3.connect(tmp_path / "telemetry.db") as connection:
        db._ensure_schema(connection)
        connection.execute("UPDATE schema_meta SET value = 'future-version' WHERE key='version'")
        connection.commit()
        db._ensure_schema(connection)
        assert connection.execute("SELECT value FROM schema_meta WHERE key='version'").fetchone() == ("future-version",)
