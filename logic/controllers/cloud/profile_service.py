"""The service for a cloud profile (``cloud --profile NAME ...``, CR-17).

A profile (:mod:`.auth.config`) names a control plane, an object store, a provider with
a default compute kind and region, and how to authenticate. :func:`profile_service`
builds a :class:`.service.CloudService` that submits to that control plane, stages
inputs and reads results through that store, and starts tasks on the provider's
backends (every compute kind it has, plus the local process backend). Constructor
arguments a backend does not take are left out, so the same call serves GCP, AWS and
Azure; ``backend_options`` in the profile passes provider settings (project, queue,
resource group; never secrets) to the backends that accept them.

The cost, budget and pool settings come from the local state directory
(``<root>/prices.json``, ``budget.json``, ``pool.json``), as for the local profile.

Attributes:
    profile_service: The service for a profile.
    backend_kwargs: Keyword arguments a backend class accepts.
    gcp_project_id: The project id a profile already names.
    bind_store_project: Attach that project to a GCS store.
    store_for_profile: The profile's store with that project attached.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

from logic.controllers.cloud.model import ComputeKind, Provider


def backend_kwargs(cls: type, offered: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of ``offered`` that ``cls.__init__`` takes (all of it if it takes ``**kwargs``)."""
    init = cast(Callable[..., Any], cast(Any, cls).__init__)
    parameters = inspect.signature(init).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(offered)
    return {k: v for k, v in offered.items() if k in parameters and v is not None}


def gcp_project_id(profile: Any) -> Optional[str]:
    """The GCP project a profile already names.

    ``backend_options.project`` wins. A ``firestore://<project>`` control plane
    names the same project when that option is absent. User Application Default
    Credentials often carry neither a project nor a quota project, and Studio
    does not set ``GOOGLE_CLOUD_PROJECT``.
    """
    from urllib.parse import urlparse

    options = getattr(profile, "backend_options", None) or {}
    project = options.get("project")
    if isinstance(project, str) and project.strip():
        return project.strip()
    parsed = urlparse(str(getattr(profile, "control_plane", "") or ""))
    if parsed.scheme == "firestore" and parsed.netloc:
        return parsed.netloc
    return None


def bind_store_project(store: Any, project: Optional[str]) -> None:
    """Point a GCS store at ``project`` so its client can be built from user ADC."""
    if project and hasattr(store, "project"):
        store.project = project


def store_for_profile(profile: Any) -> Any:
    """The profile's object store, with its GCP project attached when it has one."""
    from logic.controllers.cloud.profiles import store_from_uri

    store = store_from_uri(profile.store)
    bind_store_project(store, gcp_project_id(profile))
    return store


def backends_for_profile(profile: Any) -> Dict[Any, Any]:
    """The provider backends a profile can construct (every compute kind it accepts).

    A kind whose constructor rejects the profile's settings is left out. Local
    processes are not included; the caller adds that backend itself.
    """
    import importlib

    from logic.controllers.cloud.providers import BACKENDS

    store = store_for_profile(profile)
    offered = {
        "credentials": profile.provider_credentials(),
        "region": profile.region,
        "log_store": store,  # the task runtime uploads attempt logs here
        "status_store": store,  # self-deleting VMs write their exit codes here (GCP)
        **dict(getattr(profile, "backend_options", None) or {}),
    }
    provider = Provider(profile.provider)
    backends: Dict[Any, Any] = {}
    for (target_provider, kind), target in BACKENDS.items():
        if target_provider != provider:
            continue
        module, _, name = target.partition(":")
        cls = getattr(importlib.import_module(module), name)
        try:
            backends[(target_provider, kind)] = cls(**backend_kwargs(cls, offered))
        except (TypeError, ValueError):  # a kind this profile cannot configure is simply not offered
            continue
    return backends


def profile_service(profile: Any, root: Path, project_root: Path) -> Any:
    """A :class:`.service.CloudService` for ``profile`` (see the module docstring)."""
    from logic.controllers.cloud.cost import BudgetPolicy, PriceTable
    from logic.controllers.cloud.local import LocalProcessBackend
    from logic.controllers.cloud.local.object_store import EnvSecretStore
    from logic.controllers.cloud.profiles import control_plane_from_uri
    from logic.controllers.cloud.providers import BACKENDS
    from logic.controllers.cloud.service import CloudService

    root = Path(root)
    cp = control_plane_from_uri(profile.control_plane)
    store = store_for_profile(profile)
    credentials = profile.provider_credentials()
    offered = {
        "credentials": credentials,
        "region": profile.region,
        "log_store": store,  # the task runtime uploads attempt logs here
        "status_store": store,  # self-deleting VMs write their exit codes here (GCP)
        **dict(getattr(profile, "backend_options", None) or {}),
    }
    import importlib

    provider = Provider(profile.provider)
    skipped: Dict[str, str] = {}
    backends: Dict[Any, Any] = {
        (Provider.LOCAL, ComputeKind.PROCESS): LocalProcessBackend(
            cwd=project_root, secrets=EnvSecretStore(prefix="WSR_SECRET_")
        )
    }
    for (target_provider, kind), target in BACKENDS.items():
        if target_provider != provider:
            continue
        module, _, name = target.partition(":")
        cls = getattr(importlib.import_module(module), name)
        try:
            backends[(target_provider, kind)] = cls(**backend_kwargs(cls, offered))
        except (TypeError, ValueError) as exc:  # a kind this profile cannot configure is simply not offered
            skipped[kind.value] = str(exc)

    def predictor():
        from logic.controllers.cloud.prediction import fitted_predictor

        return fitted_predictor(cp)

    budget_file, pool_file = root / "budget.json", root / "pool.json"
    pool = None
    if pool_file.is_file():
        from logic.controllers.cloud.cli import parse_target
        from logic.controllers.cloud.scheduler import PoolPolicy

        data = json.loads(pool_file.read_text())
        fields = cast(Dict[str, Any], {**data, "target": parse_target(data["target"])})
        pool = PoolPolicy(**fields)
    service = CloudService(
        control_plane=cp,
        store=store,
        backends=backends,
        workroot=root / "work",
        remote=profile.remote_config(project_root),
        predictor_factory=predictor,
        prices=PriceTable(root / "prices.json"),
        budget=BudgetPolicy(**json.loads(budget_file.read_text())) if budget_file.is_file() else None,
        pool=pool,
    )
    service.skipped_backends = skipped
    return service
