"""An in-memory Cloud Firestore client covering the API the Firebase control plane uses.

It enforces the rules the real service enforces, so code that would fail on
Firestore fails here too:

- document ids may not contain ``/`` and may not be ``.`` or ``..``;
- an array may not directly contain another array;
- a document is at most 1 MiB;
- a batch has at most 500 writes;
- in a transaction, every read comes before the first write;
- a query with an inequality filter must order by that field first.

Transactions are serialised with one lock, which is enough for the
exclusive-lease contract (real Firestore retries contended transactions).

Attributes:
    FakeFirestoreClient: The client.
    FAKE_SDK: The :class:`~logic.controllers.cloud.control.firebase.FirestoreSDK` helpers for it.
"""

from __future__ import annotations

import copy
import functools
import itertools
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from logic.controllers.cloud.control.firebase import FirestoreSDK

_MAX_DOC_BYTES = 1 << 20
_OPS: Dict[str, Callable[[Any, Any], bool]] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a is not None and a < b,
    "<=": lambda a, b: a is not None and a <= b,
    ">": lambda a, b: a is not None and a > b,
    ">=": lambda a, b: a is not None and a >= b,
}
_INEQUALITY = {"<", "<=", ">", ">=", "!="}


class FirestoreError(Exception):
    """What the real client would raise (InvalidArgument, FailedPrecondition)."""


def _validate_value(value: Any, path: str, inside_array: bool = False) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise FirestoreError(f"invalid map key {key!r} at {path}")
            _validate_value(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        if inside_array:
            raise FirestoreError(f"arrays may not contain arrays ({path})")
        for index, item in enumerate(value):
            _validate_value(item, f"{path}[{index}]", inside_array=True)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise FirestoreError(f"unsupported value type {type(value).__name__} at {path}")


def _validate_doc(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise FirestoreError("a document must be a map")
    _validate_value(data, "")
    if len(json.dumps(data)) > _MAX_DOC_BYTES:
        raise FirestoreError("document exceeds 1 MiB")
    return copy.deepcopy(data)


class FakeFieldFilter:
    def __init__(self, field_path: str, op_string: str, value: Any) -> None:
        if op_string not in _OPS:
            raise FirestoreError(f"unsupported operator {op_string!r}")
        self.field_path, self.op_string, self.value = field_path, op_string, value


class FakeSnapshot:
    def __init__(self, ref: "FakeDocumentReference", data: Optional[Dict[str, Any]]) -> None:
        self.reference = ref
        self.id = ref.id
        self._data = data

    @property
    def exists(self) -> bool:
        return self._data is not None

    def to_dict(self) -> Optional[Dict[str, Any]]:
        return copy.deepcopy(self._data)


class FakeDocumentReference:
    def __init__(self, client: "FakeFirestoreClient", collection: str, doc_id: str) -> None:
        if not doc_id or "/" in doc_id or doc_id in (".", "..") or len(doc_id.encode()) > 1500:
            raise FirestoreError(f"invalid document id {doc_id!r}")
        self._client, self._collection, self.id = client, collection, doc_id

    def get(self, transaction: Optional["FakeTransaction"] = None) -> FakeSnapshot:
        if transaction is not None:
            transaction._read()
        with self._client._lock:
            return FakeSnapshot(self, self._client._data.get(self._collection, {}).get(self.id))

    def set(self, data: Dict[str, Any]) -> None:
        doc = _validate_doc(data)
        with self._client._lock:
            self._client._data.setdefault(self._collection, {})[self.id] = doc

    def delete(self) -> None:
        with self._client._lock:
            self._client._data.get(self._collection, {}).pop(self.id, None)


class FakeQuery:
    def __init__(self, client: "FakeFirestoreClient", collection: str) -> None:
        self._client, self._collection = client, collection
        self._filters: List[FakeFieldFilter] = []
        self._orders: List[Tuple[str, bool]] = []
        self._limit: Optional[int] = None

    def _copy(self) -> "FakeQuery":
        query = FakeQuery(self._client, self._collection)
        query._filters, query._orders, query._limit = list(self._filters), list(self._orders), self._limit
        return query

    def where(self, *args: Any, filter: Optional[FakeFieldFilter] = None) -> "FakeQuery":  # noqa: A002 - SDK name
        if filter is None:
            filter = FakeFieldFilter(*args)  # noqa: A001
        query = self._copy()
        query._filters.append(filter)
        return query

    def order_by(self, field_path: str, direction: str = "ASCENDING") -> "FakeQuery":
        query = self._copy()
        query._orders.append((field_path, direction == "DESCENDING"))
        return query

    def limit(self, count: int) -> "FakeQuery":
        query = self._copy()
        query._limit = count
        return query

    def stream(self, transaction: Optional["FakeTransaction"] = None):
        inequality = {f.field_path for f in self._filters if f.op_string in _INEQUALITY}
        if len(inequality) > 1:
            raise FirestoreError("inequality filters on more than one field")
        if inequality and self._orders and self._orders[0][0] not in inequality:
            raise FirestoreError("the first order_by must be on the inequality filter's field")
        if transaction is not None:
            transaction._read()
        with self._client._lock:
            items = list(self._client._data.get(self._collection, {}).items())
        rows = [
            (doc_id, doc)
            for doc_id, doc in items
            if all(_OPS[f.op_string](doc.get(f.field_path), f.value) for f in self._filters)
        ]
        for field_path, descending in reversed(self._orders):
            rows.sort(key=lambda row: (row[1].get(field_path) is None, row[1].get(field_path)), reverse=descending)
        if self._limit is not None:
            rows = rows[: self._limit]
        return [
            FakeSnapshot(FakeDocumentReference(self._client, self._collection, doc_id), copy.deepcopy(doc))
            for doc_id, doc in rows
        ]

    def get(self, transaction: Optional["FakeTransaction"] = None):
        return self.stream(transaction=transaction)


class FakeCollection(FakeQuery):
    def document(self, doc_id: str) -> FakeDocumentReference:
        return FakeDocumentReference(self._client, self._collection, doc_id)

    def add(self, data: Dict[str, Any]):
        ref = self.document(f"auto{next(self._client._ids):012d}")
        ref.set(data)
        return None, ref


class FakeWriteBatch:
    def __init__(self, client: "FakeFirestoreClient") -> None:
        self._client = client
        self._writes: List[Tuple[str, FakeDocumentReference, Optional[Dict[str, Any]]]] = []

    def set(self, ref: FakeDocumentReference, data: Dict[str, Any]) -> None:
        self._writes.append(("set", ref, _validate_doc(data)))

    def delete(self, ref: FakeDocumentReference) -> None:
        self._writes.append(("delete", ref, None))

    def commit(self) -> None:
        if len(self._writes) > 500:
            raise FirestoreError("a batch may contain at most 500 writes")
        with self._client._lock:
            for op, ref, data in self._writes:
                if op == "set" and data is not None:
                    ref.set(data)
                else:
                    ref.delete()
        self._writes = []


class FakeTransaction(FakeWriteBatch):
    def __init__(self, client: "FakeFirestoreClient") -> None:
        super().__init__(client)

    def _read(self) -> None:
        if self._writes:
            raise FirestoreError("transactions require all reads to be executed before all writes")

    def update(self, ref: FakeDocumentReference, data: Dict[str, Any]) -> None:
        current = ref.get().to_dict()
        if current is None:
            raise FirestoreError(f"no document to update: {ref.id}")
        current.update(data)
        self.set(ref, current)


class FakeFirestoreClient:
    """In-memory stand-in for ``google.cloud.firestore.Client``."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._ids = itertools.count()

    def collection(self, name: str) -> FakeCollection:
        return FakeCollection(self, name)

    def batch(self) -> FakeWriteBatch:
        return FakeWriteBatch(self)

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)


def fake_transactional(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Like ``google.cloud.firestore.transactional``: run ``fn(transaction)`` and commit its writes."""

    @functools.wraps(fn)
    def run(transaction: FakeTransaction, *args: Any, **kwargs: Any) -> Any:
        with transaction._client._lock:
            result = fn(transaction, *args, **kwargs)
            transaction.commit()
            return result

    return run


FAKE_SDK = FirestoreSDK(transactional=fake_transactional, field_filter=FakeFieldFilter, descending="DESCENDING")
