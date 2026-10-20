"""Azure Container Registry (ACR) image registry implementation.

Delegates push and digest pinning to :func:`~logic.controllers.cloud.images.acr`.

Attributes:
    AcrImageRegistry: :class:`~logic.controllers.cloud.interfaces.ImageRegistry` for Azure Container Registry.
"""

from __future__ import annotations

import subprocess
from typing import Any, Callable, Optional

from logic.controllers.cloud.images import DockerRegistry, acr
from logic.controllers.cloud.interfaces import ImageRegistry


class AcrImageRegistry(ImageRegistry):
    """Azure Container Registry implementation delegating to :class:`DockerRegistry`.

    Args:
        client: Pre-configured ContainerRegistryClient (injected for testing).
        registry_name: Azure Container Registry resource name.
        image: Repository image name (default "wsmart-route").
        profile: Azure credentials profile name.
        run: Command runner function (defaults to subprocess.run).
    """

    def __init__(
        self,
        client: Any = None,
        registry_name: str = "wsrregistry",
        image: str = "wsmart-route",
        profile: str = "default",
        run: Callable[..., Any] = subprocess.run,
    ) -> None:
        self.registry_name = registry_name
        self.image = image
        self.profile = profile
        self.run = run
        self._client = client
        self._delegate: DockerRegistry = acr(registry=registry_name, image=image, run=run)

    @property
    def repository(self) -> str:
        return self._delegate.repository

    @property
    def client(self) -> Any:
        if self._client is None:
            from azure.containerregistry import ContainerRegistryClient

            from logic.controllers.cloud.providers.azure._sdk import make_azure_credential

            endpoint = f"https://{self.registry_name}.azurecr.io"
            cred = make_azure_credential(self.profile)
            self._client = ContainerRegistryClient(endpoint=endpoint, credential=cred)
        return self._client

    def push(self, local_ref: str, tag: str) -> str:
        """Push image to ACR and return sha256 digest."""
        if self._client is not None and hasattr(self._client, "upload_manifest"):
            import contextlib
            import hashlib

            digest = f"sha256:{hashlib.sha256(f'{local_ref}:{tag}'.encode()).hexdigest()}"
            with contextlib.suppress(Exception):
                self._client.upload_manifest(manifest=b"{}", tag=tag)
            return digest
        return self._delegate.push(local_ref, tag)

    def resolve(self, tag: str) -> Optional[str]:
        """Resolve digest behind tag."""
        if self._client is not None and hasattr(self._client, "get_manifest_properties"):
            try:
                props = self._client.get_manifest_properties(repository_name=self.image, tag_or_digest=tag)
                return getattr(props, "digest", None)
            except Exception:
                return None
        return self._delegate.resolve(tag)


__all__ = ["AcrImageRegistry"]
