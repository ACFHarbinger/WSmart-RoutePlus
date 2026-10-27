"""Lazy Azure SDK client factories.

Azure SDKs are imported strictly inside functions to keep the core package independent
of provider dependencies.
"""

from __future__ import annotations

from typing import Any, Optional


class MsrestTokenCredentialAdapter:
    """Adapter wrapping an Azure Identity TokenCredential for msrest-based SDKs (e.g. Azure Batch 14.x)."""

    def __init__(
        self,
        token_credential: Any,
        resource: str = "https://batch.core.windows.net/",
    ) -> None:
        self._cred = token_credential
        self._resource = resource

    def signed_session(self, session: Any = None) -> Any:
        import requests  # type: ignore[import-untyped]  # requests has no stubs

        session = session or requests.Session()
        scope = self._resource if self._resource.endswith("/.default") else f"{self._resource.rstrip('/')}/.default"
        token = self._cred.get_token(scope).token
        session.headers["Authorization"] = f"Bearer {token}"
        return session

    def refresh_session(self, session: Any = None) -> Any:
        return self.signed_session(session)


def make_azure_credential(profile: str = "default", credentials: Optional[Any] = None) -> Any:
    """Create Azure credential instance lazily.

    Args:
        profile: Profile / account identifier.
        credentials: Optional ProviderCredentials or existing credential instance.

    Returns:
        azure.identity credential instance.
    """
    if credentials is not None:
        if hasattr(credentials, "azure_credential"):
            return credentials.azure_credential()
        return credentials

    if profile and profile != "default":
        try:
            from logic.controllers.cloud.auth.credentials import ProviderCredentials

            return ProviderCredentials(provider="azure", profile=profile).azure_credential()
        except Exception:
            pass

    from azure.identity import DefaultAzureCredential  # pyrefly: ignore[missing-import]  # optional Azure SDK

    return DefaultAzureCredential()


def make_blob_service_client(
    account_url: Optional[str] = None,
    credential: Any = None,
    credentials: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure BlobServiceClient lazily."""
    from azure.storage.blob import BlobServiceClient  # pyrefly: ignore[missing-import]  # optional Azure SDK

    cred = credential or credentials
    if cred is not None and hasattr(cred, "azure_credential"):
        cred = cred.azure_credential()
    elif cred is None:
        cred = make_azure_credential(profile)

    url = account_url or "https://wsrstorage.blob.core.windows.net"
    return BlobServiceClient(account_url=url, credential=cred)


def make_secret_client(
    vault_url: str,
    credential: Any = None,
    credentials: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure KeyVault SecretClient lazily."""
    from azure.keyvault.secrets import SecretClient  # pyrefly: ignore[missing-import]  # optional Azure SDK

    cred = credential or credentials
    if cred is not None and hasattr(cred, "azure_credential"):
        cred = cred.azure_credential()
    elif cred is None:
        cred = make_azure_credential(profile)

    return SecretClient(vault_url=vault_url, credential=cred)


def make_batch_service_client(
    batch_url: str,
    credential: Any = None,
    credentials: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure BatchServiceClient lazily."""
    from azure.batch import BatchServiceClient  # pyrefly: ignore[missing-import]  # optional Azure SDK

    cred = credential or credentials
    if cred is not None and hasattr(cred, "azure_credential"):
        cred = cred.azure_credential()
    elif cred is None:
        cred = make_azure_credential(profile)

    if cred is not None and not hasattr(cred, "signed_session"):
        cred = MsrestTokenCredentialAdapter(cred)

    return BatchServiceClient(credentials=cred, batch_url=batch_url)


def make_container_apps_client(
    subscription_id: str,
    credential: Any = None,
    credentials: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure ContainerAppsAPIClient lazily."""
    from azure.mgmt.appcontainers import ContainerAppsAPIClient  # pyrefly: ignore[missing-import]  # optional Azure SDK

    cred = credential or credentials
    if cred is not None and hasattr(cred, "azure_credential"):
        cred = cred.azure_credential()
    elif cred is None:
        cred = make_azure_credential(profile)

    return ContainerAppsAPIClient(credential=cred, subscription_id=subscription_id)


def make_compute_management_client(
    subscription_id: str,
    credential: Any = None,
    credentials: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure ComputeManagementClient lazily."""
    from azure.mgmt.compute import ComputeManagementClient  # pyrefly: ignore[missing-import]  # optional Azure SDK

    cred = credential or credentials
    if cred is not None and hasattr(cred, "azure_credential"):
        cred = cred.azure_credential()
    elif cred is None:
        cred = make_azure_credential(profile)

    return ComputeManagementClient(credential=cred, subscription_id=subscription_id)


def make_network_management_client(
    subscription_id: str,
    credential: Any = None,
    credentials: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure NetworkManagementClient lazily."""
    from azure.mgmt.network import NetworkManagementClient  # pyrefly: ignore[missing-import]  # optional Azure SDK

    cred = credential or credentials
    if cred is not None and hasattr(cred, "azure_credential"):
        cred = cred.azure_credential()
    elif cred is None:
        cred = make_azure_credential(profile)

    return NetworkManagementClient(credential=cred, subscription_id=subscription_id)


__all__ = [
    "MsrestTokenCredentialAdapter",
    "make_azure_credential",
    "make_batch_service_client",
    "make_blob_service_client",
    "make_compute_management_client",
    "make_container_apps_client",
    "make_network_management_client",
    "make_secret_client",
]
