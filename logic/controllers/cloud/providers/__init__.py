"""Provider compute backends, object stores, secret stores, and registries.

Attributes:
    BACKENDS: Mapping from (Provider, ComputeKind) to "module:ClassName" string
        for lazy instantiation without importing provider SDKs.
    load_backend: Dynamically resolve and instantiate a compute backend.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from logic.controllers.cloud.interfaces import ComputeBackend
from logic.controllers.cloud.model import ComputeKind, Provider

BACKENDS: Dict[Tuple[Provider, ComputeKind], str] = {
    (Provider.GCP, ComputeKind.BATCH): "logic.controllers.cloud.providers.gcp.batch:GcpBatchBackend",
    (Provider.GCP, ComputeKind.SERVERLESS): "logic.controllers.cloud.providers.gcp.serverless:CloudRunJobsBackend",
    (Provider.GCP, ComputeKind.VM): "logic.controllers.cloud.providers.gcp.vm:ComputeEngineBackend",
    (Provider.AWS, ComputeKind.BATCH): "logic.controllers.cloud.providers.aws.batch:AwsBatchBackend",
    (Provider.AWS, ComputeKind.SERVERLESS): "logic.controllers.cloud.providers.aws.serverless:EcsFargateBackend",
    (Provider.AWS, ComputeKind.VM): "logic.controllers.cloud.providers.aws.vm:Ec2Backend",
    (Provider.AZURE, ComputeKind.BATCH): "logic.controllers.cloud.providers.azure.batch:AzureBatchBackend",
    (Provider.AZURE, ComputeKind.SERVERLESS): "logic.controllers.cloud.providers.azure.serverless:ContainerAppsJobsBackend",
    (Provider.AZURE, ComputeKind.VM): "logic.controllers.cloud.providers.azure.vm:AzureVmBackend",
}


def load_backend(provider: Provider, kind: ComputeKind, **kwargs: Any) -> ComputeBackend:
    """Resolve and instantiate a backend dynamically.

    Args:
        provider: Cloud provider.
        kind: Compute service kind.
        **kwargs: Constructor arguments passed to backend.

    Returns:
        Instantiated ComputeBackend.

    Raises:
        ValueError: If provider/kind pair is not registered in BACKENDS.
    """
    key = (provider, kind)
    if key not in BACKENDS:
        available = ", ".join(f"{p.value}/{k.value}" for p, k in BACKENDS)
        raise ValueError(f"no backend registered for {provider.value}/{kind.value} (available: {available})")

    target = BACKENDS[key]
    module_name, class_name = target.split(":")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


__all__ = ["BACKENDS", "load_backend"]
