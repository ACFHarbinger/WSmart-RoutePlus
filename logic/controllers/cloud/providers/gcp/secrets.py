"""Google Secret Manager secret store (CR-05).

Secret ids allow letters, digits, ``-`` and ``_``; other characters in a name (for
example the ``/`` of ``gcp/default`` credential accounts) become ``--``.

Attributes:
    GcpSecretManagerStore: :class:`~logic.controllers.cloud.interfaces.SecretStore` on Secret Manager.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from logic.controllers.cloud.interfaces import SecretStore
from logic.controllers.cloud.providers.gcp._sdk import client as make_client
from logic.controllers.cloud.providers.gcp._sdk import field


def _not_found(exc: Exception) -> bool:
    return isinstance(exc, KeyError) or getattr(exc, "code", None) == 404 or type(exc).__name__ == "NotFound"


class GcpSecretManagerStore(SecretStore):
    """Secrets ``projects/<project>/secrets/<prefix><name>``, read at their latest version.

    Args:
        project: Project id.
        prefix: Secret id prefix.
        client: A ``SecretManagerServiceClient`` (tests inject a fake).
        credentials: ProviderCredentials or google.auth credentials (default: ADC).
    """

    def __init__(self, project: str, prefix: str = "", client: Any = None, credentials: Any = None) -> None:
        self.project, self.prefix, self.credentials = project, prefix, credentials
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client("secret_manager", self.credentials)
        return self._client

    def secret_id(self, name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]", "--", f"{self.prefix}{name}")

    def resource(self, name: str) -> str:
        """``projects/<project>/secrets/<id>``; a full resource name is returned as given."""
        return name if name.startswith("projects/") else f"projects/{self.project}/secrets/{self.secret_id(name)}"

    def version(self, name: str) -> str:
        """The latest version's resource name (what Batch and Cloud Run reference)."""
        return f"{self.resource(name)}/versions/latest"

    def get(self, name: str) -> Optional[str]:
        try:
            response = self.client.access_secret_version(name=self.version(name))
        except Exception as exc:
            if _not_found(exc):
                return None
            raise
        data = field(response, "payload.data", b"")
        return data.decode() if isinstance(data, (bytes, bytearray)) else str(data)

    def put(self, name: str, value: str) -> None:
        try:
            self.client.create_secret(
                parent=f"projects/{self.project}",
                secret_id=self.secret_id(name),
                secret={"replication": {"automatic": {}}},
            )
        except Exception as exc:
            if type(exc).__name__ != "AlreadyExists" and getattr(exc, "code", None) != 409:
                raise
        self.client.add_secret_version(parent=self.resource(name), payload={"data": value.encode()})

    def delete(self, name: str) -> None:
        try:
            self.client.delete_secret(name=self.resource(name))
        except Exception as exc:
            if not _not_found(exc):
                raise


__all__ = ["GcpSecretManagerStore"]
