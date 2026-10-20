"""Google Cloud Run Jobs backend (CR-05).

Each attempt creates a Cloud Run job ``wsr-<task>-a<n>`` with one task and no Cloud Run
retries, then runs one execution of it; the handle is the execution. Cloud Run limits a
task to 24 hours, 8 vCPUs and 32 GiB, so a launch beyond them raises
:class:`CapacityError` (the scheduler places it elsewhere). There is no spot capacity.
``secret_env`` becomes environment variables sourced from Secret Manager. The job is
deleted once its execution has finished.

Attributes:
    CloudRunJobsBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for Cloud Run Jobs.
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

from logic.controllers.cloud.interfaces import (
    AttemptState,
    BackendCapabilities,
    BackendStatus,
    CapacityError,
    ComputeBackend,
    LaunchSpec,
)
from logic.controllers.cloud.model import ComputeKind, Provider
from logic.controllers.cloud.providers.gcp._common import labels, raise_capacity, resource_name, runtime_log
from logic.controllers.cloud.providers.gcp._sdk import client as make_client
from logic.controllers.cloud.providers.gcp._sdk import field
from logic.controllers.cloud.providers.gcp.pricing import price_per_hour
from logic.controllers.cloud.providers.gcp.secrets import GcpSecretManagerStore

MAX_TASK_SECONDS = 86400
MAX_CPUS, MAX_MEMORY_GB = 8.0, 32.0


def _not_found(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 404 or type(exc).__name__ == "NotFound" or isinstance(exc, KeyError)


def _result(operation: Any) -> Any:
    """Wait for a long-running operation when the client returns one."""
    return operation.result() if callable(getattr(operation, "result", None)) else operation


class CloudRunJobsBackend(ComputeBackend):
    """Cloud Run jobs in ``projects/<project>/locations/<region>``.

    Args:
        project: Project id.
        region: Cloud Run region.
        jobs_client: A ``run_v2.JobsClient`` (tests inject a fake).
        executions_client: A ``run_v2.ExecutionsClient`` (tests inject a fake).
        secrets: Secret Manager store resolving ``secret_env`` names.
        log_store: Object store with the runtime's uploaded attempt logs.
        service_account: Service account the job runs as.
        credentials: ProviderCredentials or google.auth credentials (default: ADC).
    """

    provider = Provider.GCP
    kind = ComputeKind.SERVERLESS

    def __init__(
        self,
        project: str = "wsr-project",
        region: str = "europe-west1",
        jobs_client: Any = None,
        executions_client: Any = None,
        secrets: Optional[GcpSecretManagerStore] = None,
        log_store: Any = None,
        service_account: Optional[str] = None,
        credentials: Any = None,
    ) -> None:
        self.project, self.region = project, region
        self.secrets = secrets or GcpSecretManagerStore(project, credentials=credentials)
        self.log_store, self.service_account, self.credentials = log_store, service_account, credentials
        self._jobs, self._executions = jobs_client, executions_client
        self._labels: Dict[str, Dict[str, str]] = {}

    @property
    def jobs(self) -> Any:
        if self._jobs is None:
            self._jobs = make_client("run_jobs", self.credentials)
        return self._jobs

    @property
    def executions(self) -> Any:
        if self._executions is None:
            self._executions = make_client("run_executions", self.credentials)
        return self._executions

    @property
    def parent(self) -> str:
        return f"projects/{self.project}/locations/{self.region}"

    def capabilities(self) -> BackendCapabilities:
        sizes = {
            "cloudrun.1vcpu-2gib": {"cpus": 1.0, "memory_gb": 2.0},
            "cloudrun.2vcpu-4gib": {"cpus": 2.0, "memory_gb": 4.0},
        }
        sizes["cloudrun.4vcpu-8gib"] = {"cpus": 4.0, "memory_gb": 8.0}
        return BackendCapabilities(
            provider=Provider.GCP,
            kinds=[ComputeKind.SERVERLESS],
            max_wall_seconds=MAX_TASK_SECONDS,
            supports_spot=False,
            regions=[self.region],
            machine_types=sizes,
        )

    def job(self, launch: LaunchSpec) -> Dict[str, Any]:
        """The Cloud Run ``Job`` resource for ``launch`` (a dict the client accepts)."""
        r = launch.resources
        env = [{"name": k, "value": str(v)} for k, v in sorted(launch.env.items())]
        env += [
            {
                "name": var,
                "value_source": {"secret_key_ref": {"secret": self.secrets.resource(name), "version": "latest"}},
            }
            for var, name in sorted(launch.secret_env.items())
        ]
        container = {
            "image": launch.image or "",
            "command": [launch.argv[0]],
            "args": list(launch.argv[1:]),
            "env": env,
            "resources": {"limits": {"cpu": f"{r.cpus:g}", "memory": f"{int(round(r.memory_gb * 1024))}Mi"}},
        }
        task: Dict[str, Any] = {"containers": [container], "max_retries": 0}
        task["timeout"] = f"{int(min(r.max_wall_seconds or MAX_TASK_SECONDS, MAX_TASK_SECONDS))}s"
        if self.service_account:
            task["service_account"] = self.service_account
        return {"template": {"task_count": 1, "parallelism": 1, "template": task}, "labels": labels(launch)}

    def submit(self, launch: LaunchSpec) -> str:
        r = launch.resources
        if r.max_wall_seconds is not None and r.max_wall_seconds > MAX_TASK_SECONDS:
            raise CapacityError(f"Cloud Run tasks run at most {MAX_TASK_SECONDS}s; {r.max_wall_seconds:.0f}s requested")
        if r.cpus > MAX_CPUS or r.memory_gb > MAX_MEMORY_GB:
            raise CapacityError(f"Cloud Run tasks have at most {MAX_CPUS:g} vCPUs and {MAX_MEMORY_GB:g} GiB")
        name = resource_name(launch)
        try:
            _result(self.jobs.create_job(parent=self.parent, job=self.job(launch), job_id=name))
            operation = self.jobs.run_job(name=f"{self.parent}/jobs/{name}")
        except Exception as exc:
            raise_capacity(exc)
            raise
        execution = field(operation, "metadata.name") or field(_result(operation), "name")
        self._labels[execution] = labels(launch)
        return f"gcp-run:{execution}"

    @staticmethod
    def _name(handle: str) -> str:
        return handle.split(":", 1)[1] if handle.startswith("gcp-run:") else handle

    def poll(self, handle: str) -> BackendStatus:
        try:
            execution = self.executions.get_execution(name=self._name(handle))
        except Exception as exc:
            return BackendStatus(AttemptState.UNKNOWN, message="no such attempt" if _not_found(exc) else str(exc))
        conditions = [
            str(field(c, "message", "")) for c in field(execution, "conditions", []) or [] if field(c, "message")
        ]
        message = conditions[-1] if conditions else None
        if int(field(execution, "succeeded_count", 0)) >= 1:
            self._cleanup(handle)
            return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)
        if int(field(execution, "cancelled_count", 0)) >= 1:
            self._cleanup(handle)
            return BackendStatus(AttemptState.FAILED, exit_code=143, message="cancelled")
        if int(field(execution, "failed_count", 0)) >= 1:
            self._cleanup(handle)
            return BackendStatus(AttemptState.FAILED, exit_code=1, message=message or "task failed")
        if int(field(execution, "running_count", 0)) >= 1:
            return BackendStatus(AttemptState.RUNNING, message=message)
        return BackendStatus(AttemptState.PENDING, message=message)

    def _cleanup(self, handle: str) -> None:
        job = self._name(handle).split("/executions/")[0]
        with contextlib.suppress(Exception):
            self.jobs.delete_job(name=job)

    def cancel(self, handle: str) -> None:
        with contextlib.suppress(Exception):
            self.executions.cancel_execution(name=self._name(handle))

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        execution = self._name(handle)
        resource_labels = self._labels.get(execution)
        if resource_labels is None:
            with contextlib.suppress(Exception):
                resource_labels = dict(field(self.executions.get_execution(name=execution), "labels", {}) or {})
        return runtime_log(self.log_store, resource_labels or {}, tail)

    def price_per_hour(self, machine_type: Optional[str], region: Optional[str], spot: bool) -> Optional[float]:
        return price_per_hour(machine_type or "cloudrun.1vcpu-2gib", region or self.region, False)


__all__ = ["CloudRunJobsBackend"]
