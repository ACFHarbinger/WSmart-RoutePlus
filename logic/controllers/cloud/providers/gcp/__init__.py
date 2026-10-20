"""Google Cloud provider (CR-05): Batch, Cloud Run Jobs and Compute Engine backends,
GCS object store, Secret Manager secret store, Artifact Registry and price hints.

No Google SDK is imported at module level; every client is created lazily
(:mod:`._sdk`) or injected (the tests use in-process fakes).
"""

from __future__ import annotations

from logic.controllers.cloud.providers.gcp.batch import GcpBatchBackend
from logic.controllers.cloud.providers.gcp.pricing import GCP_MACHINES, price_per_hour
from logic.controllers.cloud.providers.gcp.registry import ArtifactRegistry
from logic.controllers.cloud.providers.gcp.secrets import GcpSecretManagerStore
from logic.controllers.cloud.providers.gcp.serverless import CloudRunJobsBackend
from logic.controllers.cloud.providers.gcp.storage import GcsObjectStore, store_from_uri
from logic.controllers.cloud.providers.gcp.vm import ComputeEngineBackend, startup_script

__all__ = [
    "ArtifactRegistry",
    "CloudRunJobsBackend",
    "ComputeEngineBackend",
    "GCP_MACHINES",
    "GcpBatchBackend",
    "GcpSecretManagerStore",
    "GcsObjectStore",
    "price_per_hour",
    "startup_script",
    "store_from_uri",
]
