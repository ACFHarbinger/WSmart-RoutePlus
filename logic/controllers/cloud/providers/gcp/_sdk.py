"""Lazy Google Cloud client factories and response helpers.

The ``google-cloud-*`` packages are imported only inside these functions (the optional
``[cloud]`` extra). Requests are built as plain dicts, which the GAPIC clients accept for
message fields; :func:`field` reads a response whether it is an SDK message (attributes)
or a dict (the in-process fakes).

Attributes:
    client: Build one service client.
    field: Read a dotted path from a message or dict.
    state_name: An enum or string state as its upper-case name.
"""

from __future__ import annotations

import importlib
from typing import Any, Optional

# service -> (module, class) of the GAPIC client
_CLIENTS = {
    "batch": ("google.cloud.batch_v1", "BatchServiceClient"),
    "run_jobs": ("google.cloud.run_v2", "JobsClient"),
    "run_executions": ("google.cloud.run_v2", "ExecutionsClient"),
    "compute_instances": ("google.cloud.compute_v1", "InstancesClient"),
    "secret_manager": ("google.cloud.secretmanager", "SecretManagerServiceClient"),
}


def client(service: str, credentials: Any = None) -> Any:
    """A Google Cloud client for ``service`` (see ``_CLIENTS``), with ``credentials`` or ADC.

    ``credentials`` may be a :class:`~logic.controllers.cloud.auth.credentials.ProviderCredentials`
    (its ``gcp_credentials()`` is used) or a ``google.auth`` credentials object.
    """
    creds = credentials.gcp_credentials() if hasattr(credentials, "gcp_credentials") else credentials
    if service == "storage":
        from google.cloud import storage

        return storage.Client(credentials=creds)
    module, name = _CLIENTS[service]
    cls = getattr(importlib.import_module(module), name)
    return cls(credentials=creds) if creds is not None else cls()


def field(obj: Any, path: str, default: Any = None) -> Any:
    """``obj.a.b`` or ``obj["a"]["b"]`` for ``path="a.b"``; ``default`` when missing."""
    for part in path.split("."):
        if obj is None:
            return default
        obj = obj.get(part) if isinstance(obj, dict) else getattr(obj, part, None)
    return default if obj is None else obj


def state_name(value: Any) -> str:
    """``JobStatus.State.RUNNING`` or ``"RUNNING"`` -> ``"RUNNING"``."""
    name: Optional[str] = getattr(value, "name", None)
    return (name or str(value or "")).split(".")[-1].upper()
