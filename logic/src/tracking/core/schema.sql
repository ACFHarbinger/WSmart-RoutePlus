-- schema.sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL,
    description TEXT    DEFAULT '',
    tags        TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT    PRIMARY KEY,
    experiment_id INTEGER NOT NULL,
    name          TEXT,
    status        TEXT    NOT NULL DEFAULT 'running',
    run_type      TEXT    NOT NULL DEFAULT 'generic',
    start_time    TEXT    NOT NULL,
    end_time      TEXT,
    artifact_dir  TEXT    DEFAULT '',
    error_message TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS run_tags (
    run_id TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS params (
    run_id TEXT NOT NULL,
    key    TEXT NOT NULL,
    value  TEXT NOT NULL,
    PRIMARY KEY (run_id, key),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT    NOT NULL,
    key       TEXT    NOT NULL,
    value     REAL    NOT NULL,
    step      INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics (run_id, key);

CREATE TABLE IF NOT EXISTS artifacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT    NOT NULL,
    name          TEXT    NOT NULL,
    path          TEXT    NOT NULL,
    artifact_type TEXT    DEFAULT 'file',
    file_hash     TEXT,
    size_bytes    INTEGER,
    created_at    TEXT    NOT NULL,
    metadata      TEXT    DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS dataset_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    file_path   TEXT,
    file_hash   TEXT,
    prev_hash   TEXT,
    size_bytes  INTEGER,
    shape       TEXT,
    metadata    TEXT    DEFAULT '{}',
    timestamp   TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- Lifecycle events of a run (cloud tasks: submitted, state changes, retries, results).
-- "timestamp" is the event's Unix time, so imported events keep their order.
CREATE TABLE IF NOT EXISTS run_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT    NOT NULL,
    kind      TEXT    NOT NULL,
    data      TEXT    DEFAULT '{}',
    timestamp REAL    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);
CREATE INDEX IF NOT EXISTS idx_run_events_run ON run_events (run_id, timestamp);
