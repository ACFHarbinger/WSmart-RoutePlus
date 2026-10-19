"""Firebase and Supabase control planes (CR-04) against strict in-memory fakes of their clients."""

import threading

import pytest
from logic.controllers.cloud.control.firebase import FirestoreControlPlane
from logic.controllers.cloud.control.supabase import ConflictError, SupabaseControlPlane
from logic.controllers.cloud.model import CellSpec, JobRecord, JobSpec, TaskRecord, TaskState
from logic.controllers.cloud.profiles import control_plane_from_uri
from logic.controllers.cloud.testing import control_plane_contract
from logic.controllers.cloud.testing.fake_firestore import FAKE_SDK, FakeFirestoreClient, FirestoreError
from logic.controllers.cloud.testing.fake_supabase import FakeSupabaseClient, PostgrestError


def _firestore():
    return FirestoreControlPlane(FakeFirestoreClient(), sdk=FAKE_SDK)


def _supabase():
    return SupabaseControlPlane(FakeSupabaseClient())


@pytest.mark.unit
@pytest.mark.parametrize("make", [_firestore, _supabase], ids=["firestore", "supabase"])
def test_contract(make):
    control_plane_contract(make)


def _job(cp, cells=2):
    spec = JobSpec(name="j", cells=[CellSpec(cell_id=f"a/{i}", argv=["test_sim", f"x={i}"]) for i in range(cells)])
    job = JobRecord(job_id="job-1", spec=spec, created_at=1_760_000_000.0)
    tasks = [TaskRecord(task_id=f"task-{i}", job_id="job-1", cell=c) for i, c in enumerate(spec.cells)]
    cp.create_job(job, tasks)
    return job, tasks


@pytest.mark.unit
def test_firestore_fake_rejects_what_firestore_rejects():
    client = FakeFirestoreClient()
    with pytest.raises(FirestoreError, match="document id"):
        client.collection("c").document("a/b")
    with pytest.raises(FirestoreError, match="arrays may not contain arrays"):
        client.collection("c").document("x").set({"a": [[1]]})
    with pytest.raises(FirestoreError, match="1 MiB"):
        client.collection("c").document("x").set({"a": "x" * (1 << 20)})
    transaction = client.transaction()
    ref = client.collection("c").document("y")
    transaction.set(ref, {"a": 1})
    with pytest.raises(FirestoreError, match="reads to be executed before"):
        ref.get(transaction=transaction)
    query = client.collection("c").where("a", ">", 0).order_by("b")
    with pytest.raises(FirestoreError, match="first order_by"):
        query.stream()


@pytest.mark.unit
def test_firestore_results_with_slashes_and_large_batches():
    cp = _firestore()
    _job(cp)
    from logic.controllers.cloud.model import ResultFile

    files = [ResultFile(relpath=f"30days/a/b/log_{i}.json", key=f"k/{i}", sha256="0" * 64, size=i) for i in range(600)]
    cp.record_results("task-0", files)  # more than one 500-write batch, and relpaths with "/"
    assert [f.relpath for f in cp.list_results("task-0")] == sorted(f.relpath for f in files)


@pytest.mark.unit
def test_supabase_fake_rejects_what_postgrest_rejects():
    client = FakeSupabaseClient()
    with pytest.raises(PostgrestError, match="does not exist"):
        client.table("wsr_nope").select("*").execute()
    with pytest.raises(PostgrestError, match="column"):
        client.table("wsr_jobs").select("nope").execute()
    with pytest.raises(PostgrestError, match="not-null"):
        client.table("wsr_jobs").insert({"job_id": "j"}).execute()
    client.table("wsr_jobs").insert({"job_id": "j", "created_at": 1.0, "doc": {}}).execute()
    with pytest.raises(PostgrestError, match="duplicate key"):
        client.table("wsr_jobs").insert({"job_id": "j", "created_at": 1.0, "doc": {}}).execute()
    with pytest.raises(PostgrestError, match="JSON"):
        client.table("wsr_jobs").update({"doc": {"x": object()}}).eq("job_id", "j").execute()
    with pytest.raises(PostgrestError, match="does not exist"):
        client.rpc("wsr_unknown", {}).execute()


@pytest.mark.unit
def test_supabase_insert_job_is_atomic():
    cp = _supabase()
    _job(cp)
    spec = JobSpec(name="dup", cells=[CellSpec(cell_id="c", argv=["test_sim"])])
    with pytest.raises(PostgrestError, match="duplicate key"):
        # task-0 already exists: the whole job insert must roll back
        cp.create_job(
            JobRecord(job_id="job-2", spec=spec), [TaskRecord(task_id="task-0", job_id="job-2", cell=spec.cells[0])]
        )
    with pytest.raises(KeyError):
        cp.get_job("job-2")


@pytest.mark.unit
def test_supabase_optimistic_updates_retry_on_conflict_and_never_lose_writes():
    client = FakeSupabaseClient()
    cp = SupabaseControlPlane(client)
    _job(cp)
    cp.update_task("task-0", state=TaskState.PLACING)

    # Many concurrent heartbeat writers and one state change on the same row.
    def beat(i):
        for j in range(20):
            cp.update_task("task-0", progress={f"w{i}": j})

    threads = [threading.Thread(target=beat, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    cp.update_task("task-0", state=TaskState.STARTING, backend_handle="h")
    for thread in threads:
        thread.join()
    task = cp.get_task("task-0")
    assert task.state == TaskState.STARTING and task.backend_handle == "h"
    row = client.table("wsr_tasks").select("rev,state").eq("task_id", "task-0").execute().data[0]
    assert row["state"] == "starting" and row["rev"] == task_rev(client)


def task_rev(client):
    return client.table("wsr_tasks").select("doc").eq("task_id", "task-0").execute().data[0]["doc"]["rev"]


@pytest.mark.unit
def test_supabase_gives_up_after_persistent_conflicts(monkeypatch):
    client = FakeSupabaseClient()
    cp = SupabaseControlPlane(client)
    _job(cp)
    real_table = client.table

    class Racing:
        """Every conditional update finds a newer rev, as if another writer always won."""

        def __init__(self, name):
            self._query = real_table(name)

        def __getattr__(self, attr):
            return getattr(self._query, attr)

        def update(self, values):
            client._tables["wsr_tasks"][0]["rev"] += 1
            return self._query.update(values)

    monkeypatch.setattr(client, "table", Racing)
    with pytest.raises(ConflictError):
        cp.update_task("task-0", progress={"x": 1})


@pytest.mark.unit
def test_uris_need_the_provider_sdk_or_key(monkeypatch):
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with pytest.raises(ValueError):
        control_plane_from_uri("supabase://abcdefgh")
    with pytest.raises(ValueError):
        control_plane_from_uri("firestore://my-project")
