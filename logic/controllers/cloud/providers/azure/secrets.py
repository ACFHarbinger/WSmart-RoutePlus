"""Azure Key Vault secret store implementation.

Attributes:
    KeyVaultSecretStore: :class:`~logic.controllers.cloud.interfaces.SecretStore` backed by Azure Key Vault.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any, Optional

from logic.controllers.cloud.interfaces import SecretStore
from logic.controllers.cloud.providers.azure._sdk import make_secret_client


class KeyVaultSecretStore(SecretStore):
    """Azure Key Vault secret store.

    Args:
        vault_url: Azure Key Vault URL (e.g. 'https://<vault>.vault.azure.net').
        client: Pre-configured SecretClient (injected for testing).
        profile: Azure credentials profile name.
    """

    def __init__(
        self,
        vault_url: str = "https://wsr-vault.vault.azure.net",
        client: Any = None,
        profile: str = "default",
    ) -> None:
        self.vault_url = vault_url
        self.profile = profile
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_secret_client(vault_url=self.vault_url, profile=self.profile)
        return self._client

    def _sanitize_name(self, name: str) -> str:
        """Sanitize secret name to conform to Azure Key Vault naming restrictions.

        Azure Key Vault names allow only alphanumeric characters and dashes (0-9, a-z, A-Z, -).
        """
        sanitized = re.sub(r"[^0-9a-zA-Z-]+", "-", name).strip("-")
        return sanitized or "secret"

    def get(self, name: str) -> Optional[str]:
        secret_name = self._sanitize_name(name)
        try:
            secret = self.client.get_secret(secret_name)
            return secret.value if hasattr(secret, "value") else secret.get("value")
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            code = getattr(exc, "error_code", None)
            if (
                status == 404
                or code == "SecretNotFound"
                or isinstance(exc, KeyError)
                or "not found" in str(exc).lower()
            ):
                return None
            raise

    def put(self, name: str, value: str) -> None:
        secret_name = self._sanitize_name(name)
        self.client.set_secret(secret_name, value)

    def delete(self, name: str) -> None:
        secret_name = self._sanitize_name(name)
        with contextlib.suppress(Exception):
            if hasattr(self.client, "begin_delete_secret"):
                self.client.begin_delete_secret(secret_name)
            elif hasattr(self.client, "delete_secret"):
                self.client.delete_secret(secret_name)

    def get_secret_url(self, name: str) -> str:
        """Get the full Key Vault secret URL."""
        secret_name = self._sanitize_name(name)
        base = self.vault_url.rstrip("/")
        return f"{base}/secrets/{secret_name}"


__all__ = ["KeyVaultSecretStore"]
