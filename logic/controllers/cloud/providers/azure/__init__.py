"""Azure provider package for WSmart-Route cloud runner.

Provides compute backends (Batch, Container Apps Jobs, Virtual Machines),
Blob object store, Key Vault secret store, ACR image registry, and pricing estimates.
No provider SDKs are imported at module level.
"""

from __future__ import annotations

from logic.controllers.cloud.providers.azure.batch import AzureBatchBackend
from logic.controllers.cloud.providers.azure.pricing import price_per_hour
from logic.controllers.cloud.providers.azure.registry import AcrImageRegistry
from logic.controllers.cloud.providers.azure.secrets import KeyVaultSecretStore
from logic.controllers.cloud.providers.azure.serverless import ContainerAppsJobsBackend
from logic.controllers.cloud.providers.azure.storage import BlobObjectStore
from logic.controllers.cloud.providers.azure.vm import AzureVmBackend

__all__ = [
    "AcrImageRegistry",
    "AzureBatchBackend",
    "AzureVmBackend",
    "BlobObjectStore",
    "ContainerAppsJobsBackend",
    "KeyVaultSecretStore",
    "price_per_hour",
]
