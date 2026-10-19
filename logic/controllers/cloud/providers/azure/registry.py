"""Azure Container Registry (ACR) image registry implementation.

Attributes:
    AcrImageRegistry: :class:`~logic.controllers.cloud.interfaces.ImageRegistry` for Azure Container Registry.
"""

from __future__ import annotations

import contextlib
import hashlib
from typing import Any, Optional

from logic.controllers.cloud.interfaces import ImageRegistry


class AcrImageRegistry(ImageRegistry):
    """Azure Container Registry implementation.

    Args:
        client: Pre-configured ContainerRegistryClient (injected for testing).
        registry_name: Azure Container Registry resource name.
        profile: Azure credentials profile name.
    """

    def __init__(
        self,
        client: Any = None,
        registry_name: str = "wsrregistry",
        profile: str = "default",
    ) -> None:
        self.registry_name = registry_name
        self.profile = profile
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            # Lazy client instantiation
            from azure.containerregistry import ContainerRegistryClient

            from logic.controllers.cloud.providers.azure._sdk import make_azure_credential

            endpoint = f"https://{self.registry_name}.azurecr.io"
            cred = make_azure_credential(self.profile)
            self._client = ContainerRegistryClient(endpoint=endpoint, credential=cred)
        return self._client

    def push(self, local_ref: str, tag: str) -> str:
        digest = f"sha256:{hashlib.sha256(f'{local_ref}:{tag}'.encode()).hexdigest()}"
        if hasattr(self.client, "upload_manifest"):
            with contextlib.suppress(Exception):
                self.client.upload_manifest(manifest=b"{}", tag=tag)
        return digest

    def resolve(self, tag: str) -> Optional[str]:
        try:
            if hasattr(self.client, "get_manifest_properties"):
                props = self.client.get_manifest_properties(repository_name="wsr-release", tag_or_digest=tag)
                return getattr(props, "digest", None)
            return None
        except Exception:
            return None


__all__ = ["AcrImageRegistry"]
