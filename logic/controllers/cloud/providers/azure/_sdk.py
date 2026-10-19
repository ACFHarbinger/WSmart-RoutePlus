"""Lazy Azure SDK client factories.

Azure SDKs are imported strictly inside functions to keep the core package independent
of provider dependencies.
"""

from __future__ import annotations

from typing import Any, Optional


def make_azure_credential(profile: str = "default") -> Any:
    """Create Azure credential instance lazily.

    Args:
        profile: Profile / account identifier.

    Returns:
        azure.identity credential instance.
    """
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


def make_blob_service_client(
    account_url: Optional[str] = None,
    credential: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure BlobServiceClient lazily."""
    from azure.storage.blob import BlobServiceClient

    cred = credential or make_azure_credential(profile)
    url = account_url or "https://wsrstorage.blob.core.windows.net"
    return BlobServiceClient(account_url=url, credential=cred)


def make_secret_client(
    vault_url: str,
    credential: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure KeyVault SecretClient lazily."""
    from azure.keyvault.secrets import SecretClient

    cred = credential or make_azure_credential(profile)
    return SecretClient(vault_url=vault_url, credential=cred)


def make_batch_service_client(
    batch_url: str,
    credential: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure BatchServiceClient lazily."""
    from azure.batch import BatchServiceClient

    cred = credential or make_azure_credential(profile)
    return BatchServiceClient(credentials=cred, batch_url=batch_url)


def make_container_apps_client(
    subscription_id: str,
    credential: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure ContainerAppsAPIClient lazily."""
    from azure.mgmt.appcontainers import ContainerAppsAPIClient

    cred = credential or make_azure_credential(profile)
    return ContainerAppsAPIClient(credential=cred, subscription_id=subscription_id)


def make_compute_management_client(
    subscription_id: str,
    credential: Any = None,
    profile: str = "default",
) -> Any:
    """Create Azure ComputeManagementClient lazily."""
    from azure.mgmt.compute import ComputeManagementClient

    cred = credential or make_azure_credential(profile)
    return ComputeManagementClient(credential=cred, subscription_id=subscription_id)


__all__ = [
    "make_azure_credential",
    "make_batch_service_client",
    "make_blob_service_client",
    "make_compute_management_client",
    "make_container_apps_client",
    "make_secret_client",
]
