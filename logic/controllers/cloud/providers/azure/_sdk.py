"""Lazy Azure SDK client factories.

Azure SDKs are imported strictly inside functions to keep the core package independent
of provider dependencies.
"""

from __future__ import annotations

from typing import Any, Optional


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

    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


def make_blob_service_client(
    account_url: Optional[str] = None,
    credential: Any = None,
    credentials: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure BlobServiceClient lazily."""
    from azure.storage.blob import BlobServiceClient

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
    from azure.keyvault.secrets import SecretClient

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
    from azure.batch import BatchServiceClient

    cred = credential or credentials
    if cred is not None and hasattr(cred, "azure_credential"):
        cred = cred.azure_credential()
    elif cred is None:
        cred = make_azure_credential(profile)

    return BatchServiceClient(credentials=cred, batch_url=batch_url)


def make_container_apps_client(
    subscription_id: str,
    credential: Any = None,
    credentials: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure ContainerAppsAPIClient lazily."""
    from azure.mgmt.appcontainers import ContainerAppsAPIClient

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
    from azure.mgmt.compute import ComputeManagementClient

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
    from azure.mgmt.network import NetworkManagementClient

    cred = credential or credentials
    if cred is not None and hasattr(cred, "azure_credential"):
        cred = cred.azure_credential()
    elif cred is None:
        cred = make_azure_credential(profile)

    return NetworkManagementClient(credential=cred, subscription_id=subscription_id)


__all__ = [
    "make_azure_credential",
    "make_batch_service_client",
    "make_blob_service_client",
    "make_compute_management_client",
    "make_container_apps_client",
    "make_network_management_client",
    "make_secret_client",
]
