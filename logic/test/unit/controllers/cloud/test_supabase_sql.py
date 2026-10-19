"""The Supabase migration on a real Postgres (opt-in: ``WSR_TEST_DOCKER_POSTGRES=1``, needs Docker).

Applies ``infra/cloud/supabase/migrations/0001_cloud_runner.sql`` to a throwaway
``postgres:16-alpine`` container and checks the four functions, including two
sessions leasing concurrently (``FOR UPDATE SKIP LOCKED``).
"""

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pytest

MIGRATION = Path(__file__).resolve().parents[5] / "infra/cloud/supabase/migrations/0001_cloud_runner.sql"
pytestmark = pytest.mark.skipif(
    os.environ.get("WSR_TEST_DOCKER_POSTGRES") != "1", reason="set WSR_TEST_DOCKER_POSTGRES=1 (needs Docker)"
)


@pytest.fixture(scope="module")
def psql():
    name = f"wsr-pg-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-e",
            "POSTGRES_PASSWORD=t",
            "-e",
            "POSTGRES_USER=t",
            "--memory=512m",
            "postgres:16-alpine",
        ],
        check=True,
        capture_output=True,
    )
    try:
        for _ in range(60):
            if subprocess.run(["docker", "exec", name, "pg_isready", "-U", "t"], capture_output=True).returncode == 0:
                break
            time.sleep(1)

        def run(sql: str) -> str:
            out = subprocess.run(
                ["docker", "exec", "-i", name, "psql", "-U", "t", "-q", "-v", "ON_ERROR_STOP=1", "-tA"],
                input=sql,
                capture_output=True,
                text=True,
            )
            if out.returncode:
                raise RuntimeError(out.stderr)
            return out.stdout.strip()

        run(MIGRATION.read_text())
        yield run
    finally:
        subprocess.run(["docker", "stop", name], capture_output=True)


def _insert(psql, job_id, tasks):
    job = json.dumps({"job_id": job_id, "created_at": 1.0})
    docs = json.dumps([{"task_id": t, "job_id": job_id, "seq": seq, "state": "queued", "rev": 0} for t, seq in tasks])
    psql(f"select wsr_insert_job('{job}'::jsonb, '{docs}'::jsonb);")


def test_lease_is_fifo_and_keeps_doc_and_columns_in_sync(psql):
    _insert(psql, "j1", [("t1", 2), ("t0", 1)])
    assert psql("select wsr_lease_task('w1', 100.0, 'j1')->>'task_id';") == "t0"
    assert (
        psql(
            "select doc->>'state', doc->>'lease_owner', (doc->>'rev')::int = rev, state from wsr_tasks where task_id='t0';"
        )
        == "placing|w1|t|placing"
    )
    assert json.loads(psql("select wsr_expire_leases(1000.0);")) == ["t0"]
    assert (
        psql("select state, lease_expires is null, doc->>'lease_owner' is null from wsr_tasks where task_id='t0';")
        == "queued|t|t"
    )


def test_insert_job_rolls_back_on_error(psql):
    _insert(psql, "j2", [("u0", 1)])
    with pytest.raises(RuntimeError, match="duplicate key"):
        _insert(psql, "j3", [("u0", 5)])
    assert psql("select count(*) from wsr_jobs where job_id='j3';") == "0"


def test_replace_results(psql):
    _insert(psql, "j4", [("v0", 1)])
    psql("""select wsr_replace_results('v0', '[{"relpath":"a/b.json"},{"relpath":"c.json"}]'::jsonb);""")
    psql("""select wsr_replace_results('v0', '[{"relpath":"z.json"}]'::jsonb);""")
    assert psql("select string_agg(relpath, ',') from wsr_results where task_id='v0';") == "z.json"


def test_concurrent_leases_skip_locked_rows(psql):
    _insert(psql, "j5", [("w0", 1), ("w1", 2)])
    held = {}

    def hold():
        held["a"] = psql(
            "begin; select wsr_lease_task('a', 100.0, 'j5')->>'task_id'; select pg_sleep(3); commit;"
        ).split("\n")[0]

    thread = threading.Thread(target=hold)
    thread.start()
    time.sleep(1)
    started = time.time()
    second = psql("select wsr_lease_task('b', 100.0, 'j5')->>'task_id';")
    assert time.time() - started < 2  # did not wait for the first session's lock
    thread.join()
    assert {held["a"], second} == {"w0", "w1"}
    assert psql("select wsr_lease_task('c', 100.0, 'j5');") == ""
