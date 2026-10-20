"""Azure Blob object store implementation.

Attributes:
    BlobObjectStore: :class:`~logic.controllers.cloud.interfaces.ObjectStore` backed by Azure Blob Storage.
"""

from __future__ import annotations

import contextlib
from pathlib import Path, PurePosixPath
from typing import Any, List, Optional

from logic.controllers.cloud.interfaces import ObjectStore
from logic.controllers.cloud.model import ResultFile
from logic.controllers.cloud.providers.azure._sdk import make_blob_service_client
from logic.controllers.cloud.store.storage import sha256_file


class BlobObjectStore(ObjectStore):
    """Azure Blob Storage object store.

    Args:
        container: Blob container name.
        prefix: Optional prefix for all objects in the store.
        container_client: Optional pre-configured ContainerClient (injected for testing).
        service_client: Optional pre-configured BlobServiceClient.
        profile: Azure credentials profile name.
        account_url: Storage account URL (e.g. 'https://<account>.blob.core.windows.net').
    """

    def __init__(
        self,
        container: str = "wsr-data",
        prefix: str = "",
        container_client: Any = None,
        service_client: Any = None,
        profile: str = "default",
        account_url: Optional[str] = None,
    ) -> None:
        self.container = container
        self.prefix = prefix.strip("/") + ("/" if prefix.strip("/") else "")
        self.profile = profile
        self.account_url = account_url
        self._container_client = container_client
        self._service_client = service_client

    @property
    def container_client(self) -> Any:
        if self._container_client is None:
            if self._service_client is None:
                self._service_client = make_blob_service_client(
                    account_url=self.account_url,
                    profile=self.profile,
                )
            self._container_client = self._service_client.get_container_client(self.container)
        return self._container_client

    def _resolve_key(self, key: str) -> str:
        parts = PurePosixPath(key).parts
        if not parts or key.startswith("/") or ".." in parts:
            raise ValueError(f"invalid object key {key!r}")
        return f"{self.prefix}{key}"

    def _strip_prefix(self, full_key: str) -> str:
        if self.prefix and full_key.startswith(self.prefix):
            return full_key[len(self.prefix) :]
        return full_key

    def put_file(self, local: Path, key: str) -> ResultFile:
        blob_key = self._resolve_key(key)
        local_path = Path(local)
        digest = sha256_file(local_path)
        size = local_path.stat().st_size

        blob_client = self.container_client.get_blob_client(blob_key)
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

        return ResultFile(relpath=key, key=key, sha256=digest, size=size)

    def get_file(self, key: str, local: Path) -> None:
        blob_key = self._resolve_key(key)
        target = Path(local)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            blob_client = self.container_client.get_blob_client(blob_key)
            downloader = blob_client.download_blob()
            data = downloader.readall()
            target.write_bytes(data)
        except Exception as exc:
            # Detect ResourceNotFoundError or 404 in Azure SDK or KeyError in test fakes
            status = getattr(exc, "status_code", None)
            code = getattr(exc, "error_code", None)
            if status == 404 or code == "BlobNotFound" or isinstance(exc, KeyError) or "not found" in str(exc).lower():
                raise KeyError(key) from exc
            raise

    def exists(self, key: str) -> bool:
        blob_key = self._resolve_key(key)
        try:
            blob_client = self.container_client.get_blob_client(blob_key)
            if hasattr(blob_client, "exists"):
                return bool(blob_client.exists())
            blob_client.get_blob_properties()
            return True
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            code = getattr(exc, "error_code", None)
            if status == 404 or code == "BlobNotFound" or isinstance(exc, KeyError) or "not found" in str(exc).lower():
                return False
            raise

    def list(self, prefix: str) -> List[str]:
        blob_prefix = self._resolve_key(prefix) if prefix else self.prefix
        keys: List[str] = []

        blobs = self.container_client.list_blobs(name_starts_with=blob_prefix)
        for blob in blobs:
            name = blob.name if hasattr(blob, "name") else blob.get("name", "")
            keys.append(self._strip_prefix(name))

        return sorted(keys)

    def delete(self, key: str) -> None:
        blob_key = self._resolve_key(key)
        with contextlib.suppress(Exception):
            blob_client = self.container_client.get_blob_client(blob_key)
            blob_client.delete_blob()


def store_from_uri(parsed: Any, **kwargs: Any) -> BlobObjectStore:
    """Create BlobObjectStore from a parsed az:// URI.

    URI format: az://<account>/<container>[/<prefix>]
    """
    account = parsed.netloc
    path = parsed.path.lstrip("/")
    parts = path.split("/", 1)
    container = parts[0] if parts and parts[0] else "wsr-data"
    prefix = parts[1] if len(parts) > 1 else ""
    account_url = f"https://{account}.blob.core.windows.net" if account else None
    return BlobObjectStore(
        container=container,
        prefix=prefix,
        account_url=account_url,
        **kwargs,
    )


__all__ = ["BlobObjectStore", "store_from_uri"]
