"""Google Cloud Batch backend (CR-05).

Each attempt is one Batch job with one task (``task_count: 1``): the runner's unit of
retry, resume and accounting is the task, so attempts are not grouped into task arrays.
The container runs the release image's task runtime; ``secret_env`` becomes Batch
``secret_variables`` that reference Secret Manager versions (values never pass through
the runner). Spot attempts use ``provisioning_model: SPOT``; a spot preemption (exit code
50001 or a "preempt" status event) is reported as :attr:`AttemptState.PREEMPTED`.

Attributes:
    GcpBatchBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for Cloud Batch.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any, Dict, Optional

from logic.controllers.cloud.interfaces import (
    AttemptState,
    BackendCapabilities,
    BackendStatus,
    ComputeBackend,
    LaunchSpec,
)
from logic.controllers.cloud.model import ComputeKind, Provider
from logic.controllers.cloud.providers.gcp._common import labels, raise_capacity, resource_name, runtime_log
from logic.controllers.cloud.providers.gcp._sdk import client as make_client
from logic.controllers.cloud.providers.gcp._sdk import field, state_name
from logic.controllers.cloud.providers.gcp.pricing import DEFAULT_MACHINE, GCP_MACHINES, price_per_hour
from logic.controllers.cloud.providers.gcp.secrets import GcpSecretManagerStore

PREEMPTION_EXIT_CODE = 50001
_STATES = {
    "QUEUED": AttemptState.PENDING,
    "SCHEDULED": AttemptState.PENDING,
    "RUNNING": AttemptState.RUNNING,
    "SUCCEEDED": AttemptState.SUCCEEDED,
    "FAILED": AttemptState.FAILED,
    "DELETION_IN_PROGRESS": AttemptState.FAILED,
}


def _not_found(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 404 or type(exc).__name__ == "NotFound" or isinstance(exc, KeyError)


class GcpBatchBackend(ComputeBackend):
    """Cloud Batch jobs in ``projects/<project>/locations/<region>``.

    Args:
        project: Project id.
        region: Batch location.
        client: A ``BatchServiceClient`` (tests inject a fake).
        secrets: Secret Manager store resolving ``secret_env`` names to versions.
        log_store: Object store with the runtime's uploaded attempt logs.
        service_account: Service account the job's VMs run as (default: the Compute default).
        credentials: ProviderCredentials or google.auth credentials (default: ADC).
    """

    provider = Provider.GCP
    kind = ComputeKind.BATCH

    def __init__(
        self,
        project: str = "wsr-project",
        region: str = "europe-west1",
        client: Any = None,
        secrets: Optional[GcpSecretManagerStore] = None,
        log_store: Any = None,
        service_account: Optional[str] = None,
        credentials: Any = None,
    ) -> None:
        self.project, self.region = project, region
        self.secrets = secrets or GcpSecretManagerStore(project, credentials=credentials)
        self.log_store, self.service_account, self.credentials = log_store, service_account, credentials
        self._client = client
        self._cancelled: set = set()  # deleted jobs disappear: remember them so polls report the cancel

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client("batch", self.credentials)
        return self._client

    @property
    def parent(self) -> str:
        return f"projects/{self.project}/locations/{self.region}"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            provider=Provider.GCP,
            kinds=[ComputeKind.BATCH],
            max_wall_seconds=14 * 86400,
            supports_spot=True,
            regions=[self.region],
            machine_types=dict(GCP_MACHINES),
        )

    def job(self, launch: LaunchSpec) -> Dict[str, Any]:
        """The Batch ``Job`` resource for ``launch`` (a dict the client accepts)."""
        r = launch.resources
        task_spec: Dict[str, Any] = {
            "runnables": [
                {
                    "container": {
                        "image_uri": launch.image or "",
                        "entrypoint": launch.argv[0],
                        "commands": list(launch.argv[1:]),
                    }
                }
            ],
            "environment": {
                "variables": {k: str(v) for k, v in sorted(launch.env.items())},
                "secret_variables": {
                    var: self.secrets.version(name) for var, name in sorted(launch.secret_env.items())
                },
            },
            "compute_resource": {"cpu_milli": int(round(r.cpus * 1000)), "memory_mib": int(round(r.memory_gb * 1024))},
            "max_retry_count": 0,  # the scheduler retries with resume
        }
        if r.max_wall_seconds is not None:
            task_spec["max_run_duration"] = f"{max(60, int(r.max_wall_seconds))}s"
        instance_policy = {
            "machine_type": r.machine_type or DEFAULT_MACHINE,
            "provisioning_model": "SPOT" if r.spot else "STANDARD",
        }
        allocation: Dict[str, Any] = {
            "instances": [{"policy": instance_policy}],
            "location": {"allowed_locations": [f"regions/{r.region or self.region}"]},
        }
        if self.service_account:
            allocation["service_account"] = {"email": self.service_account}
        return {
            "task_groups": [{"task_spec": task_spec, "task_count": 1, "parallelism": 1}],
            "allocation_policy": allocation,
            "logs_policy": {"destination": "CLOUD_LOGGING"},
            "labels": labels(launch),
        }

    def submit(self, launch: LaunchSpec) -> str:
        name = resource_name(launch)
        try:
            self.client.create_job(parent=self.parent, job=self.job(launch), job_id=name)
        except Exception as exc:
            raise_capacity(exc)
            raise
        return f"gcp-batch:{self.parent}/jobs/{name}"

    @staticmethod
    def _name(handle: str) -> str:
        return handle.split(":", 1)[1] if handle.startswith("gcp-batch:") else handle

    def poll(self, handle: str) -> BackendStatus:
        if handle in self._cancelled:
            return BackendStatus(AttemptState.FAILED, exit_code=143, message="cancelled")
        try:
            job = self.client.get_job(name=self._name(handle))
        except Exception as exc:
            if _not_found(exc):
                return BackendStatus(AttemptState.UNKNOWN, message="no such attempt")
            return BackendStatus(AttemptState.UNKNOWN, message=str(exc))
        state = _STATES.get(state_name(field(job, "status.state")), AttemptState.UNKNOWN)
        events = [str(field(e, "description", "")) for e in field(job, "status.status_events", []) or []]
        if state == AttemptState.FAILED:
            text = " ".join(events)
            codes = [int(c) for c in re.findall(r"exit code (\d+)", text)]
            if PREEMPTION_EXIT_CODE in codes or "preempt" in text.lower():
                return BackendStatus(
                    AttemptState.PREEMPTED, exit_code=PREEMPTION_EXIT_CODE, message="spot VM preempted"
                )
            return BackendStatus(
                AttemptState.FAILED, exit_code=codes[-1] if codes else 1, message=events[-1] if events else "job failed"
            )
        if state == AttemptState.SUCCEEDED:
            return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)
        return BackendStatus(state, message=events[-1] if events else None)

    def cancel(self, handle: str) -> None:
        if handle in self._cancelled or self.poll(handle).state not in (AttemptState.PENDING, AttemptState.RUNNING):
            return  # finished (or unknown) attempts are left alone
        with contextlib.suppress(Exception):
            self.client.delete_job(name=self._name(handle))  # Batch stops a job by deleting it
        self._cancelled.add(handle)

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        try:
            job = self.client.get_job(name=self._name(handle))
        except Exception:
            return ""
        return runtime_log(self.log_store, dict(field(job, "labels", {}) or {}), tail)

    def price_per_hour(self, machine_type: Optional[str], region: Optional[str], spot: bool) -> Optional[float]:
        return price_per_hour(machine_type, region or self.region, spot)


__all__ = ["GcpBatchBackend"]
