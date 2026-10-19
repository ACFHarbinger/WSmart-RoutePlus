"""AWS S3 object store implementation.

Attributes:
    S3ObjectStore: :class:`~logic.controllers.cloud.interfaces.ObjectStore` backed by Amazon S3.
    store_from_uri: Create S3ObjectStore from parsed s3:// URI.
"""

from __future__ import annotations

import contextlib
from pathlib import Path, PurePosixPath
from typing import Any, List, Optional

from logic.controllers.cloud.interfaces import ObjectStore
from logic.controllers.cloud.model import ResultFile
from logic.controllers.cloud.providers.aws._sdk import make_boto3_client
from logic.controllers.cloud.store.storage import sha256_file


class S3ObjectStore(ObjectStore):
    """Amazon S3 object store.

    Args:
        bucket: S3 bucket name.
        prefix: Optional prefix for all objects in the store.
        client: Optional pre-configured boto3 S3 client (injected for testing).
        credentials: Optional ProviderCredentials instance or profile name.
        profile: AWS credentials profile name (used when client is not provided).
        region: AWS region name (used when client is not provided).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client: Any = None,
        credentials: Any = None,
        profile: str = "default",
        region: Optional[str] = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/") + ("/" if prefix.strip("/") else "")
        self.credentials = credentials
        self.profile = profile
        self.region = region
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_boto3_client("s3", credentials=self.credentials, profile=self.profile, region=self.region)
        return self._client

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
        s3_key = self._resolve_key(key)
        local_path = Path(local)
        digest = sha256_file(local_path)
        size = local_path.stat().st_size

        with open(local_path, "rb") as f:
            self.client.put_object(Bucket=self.bucket, Key=s3_key, Body=f)

        return ResultFile(relpath=key, key=key, sha256=digest, size=size)

    def get_file(self, key: str, local: Path) -> None:
        s3_key = self._resolve_key(key)
        target = Path(local)
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=s3_key)
            body = resp["Body"]
            data = body.read() if hasattr(body, "read") else bytes(body)
            target.write_bytes(data)
        except Exception as exc:
            # Handle both real ClientError (NoSuchKey/404) and simulated KeyError/test fakes
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404") or isinstance(exc, KeyError) or "NoSuchKey" in str(exc):
                raise KeyError(key) from exc
            raise

    def exists(self, key: str) -> bool:
        s3_key = self._resolve_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey") or isinstance(exc, KeyError) or "404" in str(exc):
                return False
            if "NoSuchKey" in str(exc) or "not found" in str(exc).lower():
                return False
            return False

    def list(self, prefix: str) -> List[str]:
        s3_prefix = self._resolve_key(prefix) if prefix else self.prefix
        keys: List[str] = []

        # Use paginator if available, otherwise direct list_objects_v2
        if hasattr(self.client, "get_paginator"):
            try:
                paginator = self.client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
                    for item in page.get("Contents", []):
                        keys.append(self._strip_prefix(item["Key"]))
                return sorted(keys)
            except Exception:
                pass

        resp = self.client.list_objects_v2(Bucket=self.bucket, Prefix=s3_prefix)
        for item in resp.get("Contents", []):
            keys.append(self._strip_prefix(item["Key"]))

        return sorted(keys)

    def delete(self, key: str) -> None:
        s3_key = self._resolve_key(key)
        with contextlib.suppress(Exception):
            self.client.delete_object(Bucket=self.bucket, Key=s3_key)


def store_from_uri(parsed: Any, **kwargs: Any) -> S3ObjectStore:
    """Create S3ObjectStore from a parsed s3:// URI."""
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return S3ObjectStore(bucket=bucket, prefix=prefix, **kwargs)


__all__ = ["S3ObjectStore", "store_from_uri"]
