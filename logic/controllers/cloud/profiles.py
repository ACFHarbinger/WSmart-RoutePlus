"""Control planes and object stores from URIs.

A container task learns where its control plane and object store are from two
environment variables, ``WSR_CLOUD_CONTROL_PLANE`` and ``WSR_CLOUD_STORE``, which
hold URIs:

========================  =====================================================
``sqlite:///abs/path.db``  :class:`~logic.controllers.cloud.local.SQLiteControlPlane`
``file:///abs/dir``        :class:`~logic.controllers.cloud.local.FilesystemObjectStore`
``firestore://<project>``  Firebase control plane (CR-04)
``supabase://<host>``      Supabase control plane (CR-04)
``gs://<bucket>/<prefix>`` GCS object store (CR-05)
``s3://<bucket>/<prefix>`` S3 object store (CR-06)
``az://<account>/<container>/<prefix>`` Blob object store (CR-07)
========================  =====================================================

Provider schemes resolve through :data:`PROVIDER_FACTORIES`; each provider
package registers its factories there, imported lazily.

Attributes:
    control_plane_from_uri: A control plane for a URI.
    store_from_uri: An object store for a URI.
    PROVIDER_FACTORIES: ``(kind, scheme) -> "module:function"`` for provider-backed URIs.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple
from urllib.parse import urlparse

from logic.controllers.cloud.interfaces import ControlPlane, ObjectStore

# (kind, scheme) -> "module:function"; the function takes the parsed URI.
PROVIDER_FACTORIES: Dict[Tuple[str, str], str] = {
    ("control_plane", "firestore"): "logic.controllers.cloud.control.firebase:control_plane_from_uri",
    ("control_plane", "supabase"): "logic.controllers.cloud.control.supabase:control_plane_from_uri",
    ("store", "gs"): "logic.controllers.cloud.providers.gcp.storage:store_from_uri",
    ("store", "s3"): "logic.controllers.cloud.providers.aws.storage:store_from_uri",
    ("store", "az"): "logic.controllers.cloud.providers.azure.storage:store_from_uri",
}


def _provider(kind: str, uri: str):
    parsed = urlparse(uri)
    target = PROVIDER_FACTORIES.get((kind, parsed.scheme))
    if target is None:
        raise ValueError(f"unsupported {kind.replace('_', ' ')} URI {uri!r}")
    module, _, function = target.partition(":")
    try:
        factory = getattr(importlib.import_module(module), function)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"{kind.replace('_', ' ')} URI {uri!r} needs {module}, which is not available: {exc}") from exc
    try:
        return factory(parsed)
    except ImportError as exc:
        raise ValueError(
            f"{kind.replace('_', ' ')} URI {uri!r} needs a provider SDK that is not installed: {exc}"
        ) from exc


def control_plane_from_uri(uri: str) -> ControlPlane:
    """A control plane for ``uri`` (see the module table)."""
    parsed = urlparse(uri)
    if parsed.scheme == "sqlite":
        from logic.controllers.cloud.local.control_plane import SQLiteControlPlane

        return SQLiteControlPlane(parsed.path or ":memory:")
    return _provider("control_plane", uri)


def store_from_uri(uri: str) -> ObjectStore:
    """An object store for ``uri`` (see the module table)."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        from logic.controllers.cloud.local.object_store import FilesystemObjectStore

        return FilesystemObjectStore(parsed.path)
    return _provider("store", uri)
