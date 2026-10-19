"""Firebase (Cloud Firestore) control plane.

Collections: ``wsr_jobs``, ``wsr_tasks``, ``wsr_events``, ``wsr_metrics`` and
``wsr_results``. Leases and every read-modify-write run in Firestore
transactions, so concurrent schedulers and task runtimes never lose an update.
The composite indexes the queries need and the security rules are in
``infra/cloud/firebase/``.

The Firestore client and the SDK helpers (``transactional``, ``FieldFilter``,
``Query.DESCENDING``) are injected; tests pass
:class:`~logic.controllers.cloud.testing.fake_firestore.FakeFirestoreClient`.

URI: ``firestore://<project>[/<database>]`` (credentials from Application
Default Credentials or the ``gcp/<profile>`` keychain entry, CR-08).

Attributes:
    FirestoreSDK: The SDK helpers the adapter needs.
    FirestoreAdapter: :class:`~logic.controllers.cloud.control.documents.DocumentAdapter` on Firestore.
    FirestoreControlPlane: The control plane.
    control_plane_from_uri: Factory for ``firestore://`` URIs.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence
from urllib.parse import ParseResult

from logic.controllers.cloud.control.documents import (
    Doc,
    DocumentAdapter,
    DocumentControlPlane,
    Mutation,
)

JOBS, TASKS, EVENTS, METRICS, RESULTS = "wsr_jobs", "wsr_tasks", "wsr_events", "wsr_metrics", "wsr_results"
_BATCH_LIMIT = 500  # Firestore's maximum writes per batch
_APPEND_KEY = {"events": ("job_id", EVENTS), "metrics": ("task_id", METRICS)}


@dataclass
class FirestoreSDK:
    """SDK helpers: ``google.cloud.firestore.transactional``, ``FieldFilter`` and ``Query.DESCENDING``."""

    transactional: Callable[..., Any]
    field_filter: Callable[[str, str, Any], Any]
    descending: Any


def real_sdk() -> FirestoreSDK:
    """The helpers from the installed ``google-cloud-firestore``."""
    from google.cloud import firestore
    from google.cloud.firestore_v1.base_query import FieldFilter

    return FirestoreSDK(firestore.transactional, FieldFilter, firestore.Query.DESCENDING)


class FirestoreAdapter(DocumentAdapter):
    """Document primitives on Cloud Firestore."""

    def __init__(self, client: Any, sdk: Optional[FirestoreSDK] = None) -> None:
        self.client = client
        self.sdk = sdk or real_sdk()
        self._counter = itertools.count()

    def _where(self, collection: str, *clauses: tuple) -> Any:
        query = self.client.collection(collection)
        for field_name, op, value in clauses:
            query = query.where(filter=self.sdk.field_filter(field_name, op, value))
        return query

    def _commit(self, writes: Sequence[tuple]) -> None:
        for start in range(0, len(writes), _BATCH_LIMIT):
            batch = self.client.batch()
            for op, ref, data in writes[start : start + _BATCH_LIMIT]:
                if op == "set":
                    batch.set(ref, data)
                else:
                    batch.delete(ref)
            batch.commit()

    def _rmw(self, collection: str, doc_id: str, mutate: Mutation) -> Doc:
        ref = self.client.collection(collection).document(doc_id)

        @self.sdk.transactional
        def run(transaction: Any) -> Doc:
            snapshot = ref.get(transaction=transaction)
            if not snapshot.exists:
                raise KeyError(doc_id)
            updated = mutate(snapshot.to_dict())
            transaction.set(ref, updated)
            return updated

        return run(self.client.transaction())

    # Jobs and tasks --------------------------------------------------------

    def insert_job(self, job: Doc, tasks: Sequence[Doc]) -> None:
        writes = [("set", self.client.collection(JOBS).document(job["job_id"]), job)]
        writes += [("set", self.client.collection(TASKS).document(task["task_id"]), task) for task in tasks]
        self._commit(writes)

    def get_job(self, job_id: str) -> Optional[Doc]:
        snapshot = self.client.collection(JOBS).document(job_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def list_jobs(self, limit: int) -> List[Doc]:
        query = self.client.collection(JOBS).order_by("created_at", direction=self.sdk.descending).limit(limit)
        return [snapshot.to_dict() for snapshot in query.stream()]

    def update_job(self, job_id: str, mutate: Mutation) -> Doc:
        return self._rmw(JOBS, job_id, mutate)

    def get_task(self, task_id: str) -> Optional[Doc]:
        snapshot = self.client.collection(TASKS).document(task_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    def list_tasks(self, job_id: str) -> List[Doc]:
        return [s.to_dict() for s in self._where(TASKS, ("job_id", "==", job_id)).order_by("seq").stream()]

    def update_task(self, task_id: str, mutate: Mutation) -> Doc:
        return self._rmw(TASKS, task_id, mutate)

    def lease_task(self, owner: str, lease_expires: float, job_id: Optional[str]) -> Optional[Doc]:
        clauses = [("state", "==", "queued")] + ([("job_id", "==", job_id)] if job_id else [])
        query = self._where(TASKS, *clauses).order_by("seq").limit(1)

        @self.sdk.transactional
        def run(transaction: Any) -> Optional[Doc]:
            found = list(query.stream(transaction=transaction))
            if not found:
                return None
            doc = found[0].to_dict()
            doc.update(state="placing", lease_owner=owner, lease_expires=lease_expires, rev=doc.get("rev", 0) + 1)
            transaction.set(found[0].reference, doc)
            return doc

        return run(self.client.transaction())

    def expire_leases(self, now: float) -> List[str]:
        query = self._where(TASKS, ("state", "==", "placing"), ("lease_expires", "<", now))
        expired = []
        for snapshot in query.stream():

            def requeue(doc: Doc) -> Doc:
                if doc.get("state") != "placing" or (doc.get("lease_expires") or now) >= now:
                    raise _NotExpired
                return dict(doc, state="queued", lease_owner=None, lease_expires=None, rev=doc.get("rev", 0) + 1)

            try:
                self._rmw(TASKS, snapshot.id, requeue)
            except _NotExpired:
                continue
            expired.append(snapshot.id)
        return expired

    # Events, metrics, results ----------------------------------------------

    def append(self, collection: str, doc: Doc) -> None:
        _, name = _APPEND_KEY[collection]
        # ``n`` orders documents appended in the same instant; ``at`` orders across writers.
        self.client.collection(name).add(dict(doc, n=next(self._counter), written=time.time()))

    def list_appended(self, collection: str, key: str, since: Optional[float]) -> List[Doc]:
        field_name, name = _APPEND_KEY[collection]
        clauses = [(field_name, "==", key)] + ([("at", ">", since)] if since is not None else [])
        query = self._where(name, *clauses).order_by("at").order_by("n")
        docs = []
        for snapshot in query.stream():
            doc = snapshot.to_dict()
            for extra in ("n", "written"):
                doc.pop(extra, None)
            docs.append(doc)
        return docs

    def replace_results(self, task_id: str, docs: Sequence[Doc]) -> None:
        old = [("delete", s.reference, None) for s in self._where(RESULTS, ("task_id", "==", task_id)).stream()]
        new = [("set", self.client.collection(RESULTS).document(_result_id(task_id, d["relpath"])), d) for d in docs]
        self._commit(old + new)

    def list_results(self, task_id: str) -> List[Doc]:
        docs = [s.to_dict() for s in self._where(RESULTS, ("task_id", "==", task_id)).stream()]
        return sorted(docs, key=lambda doc: doc["relpath"])


def _result_id(task_id: str, relpath: str) -> str:
    """Document id of a result (relpaths contain ``/``, which Firestore ids may not)."""
    return hashlib.sha256(f"{task_id}\0{relpath}".encode()).hexdigest()


class _NotExpired(Exception):
    """Internal: the lease was renewed or released after the query ran."""


class FirestoreControlPlane(DocumentControlPlane):
    """The control plane on Cloud Firestore."""

    def __init__(self, client: Any, sdk: Optional[FirestoreSDK] = None, clock: Callable[[], float] = time.time):
        super().__init__(FirestoreAdapter(client, sdk), clock=clock)


def control_plane_from_uri(uri: ParseResult) -> FirestoreControlPlane:
    """``firestore://<project>[/<database>]``."""
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import firestore

    database = uri.path.strip("/") or "(default)"
    try:
        client = firestore.Client(project=uri.netloc, database=database)
    except DefaultCredentialsError as exc:
        raise ValueError(
            "no Google credentials for the Firestore control plane: run `gcloud auth application-default login` "
            "or use a keychain profile"
        ) from exc
    return FirestoreControlPlane(client)
