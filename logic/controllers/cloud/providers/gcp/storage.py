"""Google Cloud Storage object store (CR-05).

Attributes:
    GcsObjectStore: :class:`~logic.controllers.cloud.interfaces.ObjectStore` on a GCS bucket.
    store_from_uri: ``gs://<bucket>/<prefix>`` -> store (used by :mod:`..profiles`).
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, List, Optional

from logic.controllers.cloud.interfaces import ObjectStore
from logic.controllers.cloud.model import ResultFile
from logic.controllers.cloud.providers.gcp._sdk import client as make_client
from logic.controllers.cloud.store.storage import sha256_file


def _not_found(exc: Exception) -> bool:
    return isinstance(exc, KeyError) or getattr(exc, "code", None) == 404 or type(exc).__name__ == "NotFound"


class GcsObjectStore(ObjectStore):
    """Objects under ``gs://<bucket>/<prefix>``.

    Args:
        bucket: Bucket name.
        prefix: Key prefix inside the bucket.
        client: A ``google.cloud.storage.Client`` (tests inject a fake).
        credentials: ProviderCredentials or google.auth credentials (default: ADC).
        project: GCP project id passed to the client. Required for user ADC that
            carries no quota project (Studio's ``--json`` path does not export one).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client: Any = None,
        credentials: Any = None,
        project: Optional[str] = None,
    ) -> None:
        self.bucket_name = bucket
        self.prefix = prefix.strip("/") + ("/" if prefix.strip("/") else "")
        self.credentials = credentials
        self.project = project
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client("storage", self.credentials, project=self.project)
        return self._client

    def _blob(self, key: str) -> Any:
        parts = PurePosixPath(key).parts
        if not parts or key.startswith("/") or ".." in parts:
            raise ValueError(f"invalid object key {key!r}")
        return self.client.bucket(self.bucket_name).blob(self.prefix + key)

    def put_file(self, local: Path, key: str) -> ResultFile:
        local = Path(local)
        self._blob(key).upload_from_filename(str(local))
        return ResultFile(relpath=key, key=key, sha256=sha256_file(local), size=local.stat().st_size)

    def get_file(self, key: str, local: Path) -> None:
        target = Path(local)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._blob(key).download_to_filename(str(target))
        except Exception as exc:
            target.unlink(missing_ok=True)
            if _not_found(exc):
                raise KeyError(key) from exc
            raise

    def exists(self, key: str) -> bool:
        return bool(self._blob(key).exists())

    def list(self, prefix: str) -> List[str]:
        full = self.prefix + prefix
        names = [b.name for b in self.client.list_blobs(self.bucket_name, prefix=full)]
        return sorted(n[len(self.prefix) :] for n in names if n.startswith(self.prefix))

    def delete(self, key: str) -> None:
        try:
            self._blob(key).delete()
        except Exception as exc:
            if not _not_found(exc):
                raise


def store_from_uri(
    parsed: Any, client: Optional[Any] = None, credentials: Any = None, project: Optional[str] = None
) -> GcsObjectStore:
    """A store from a parsed ``gs://bucket/prefix`` URI."""
    return GcsObjectStore(
        bucket=parsed.netloc,
        prefix=parsed.path.lstrip("/"),
        client=client,
        credentials=credentials,
        project=project,
    )


__all__ = ["GcsObjectStore", "store_from_uri"]
