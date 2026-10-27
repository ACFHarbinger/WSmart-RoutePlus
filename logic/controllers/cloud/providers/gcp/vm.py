"""Google Compute Engine VM backend (CR-05).

Each attempt is one VM ``wsr-<task>-a<n>`` on Container-Optimized OS. Its startup script:

1. reads an access token from the metadata server;
2. runs the release image's task runtime with ``docker run`` (secrets are passed as
   ``WSR_SECRET_REFS`` references, ``gcpsm://<project>/<secret>``, which the runtime
   resolves with the VM's service account; no secret value is in the instance metadata);
3. uploads the exit code to ``<status store>/work/<job>/<task>/attempt-<n>.exit``;
4. deletes its own instance (spot VMs are also deleted on preemption).

Polling reads the instance and, once it is stopping or gone, the exit object: 0 is
success, another code a failure, and no exit object on a spot VM a preemption.

Attributes:
    ComputeEngineBackend: :class:`~logic.controllers.cloud.interfaces.ComputeBackend` for VMs.
    startup_script: The VM's startup script.
"""

from __future__ import annotations

import contextlib
import json
import shlex
import tempfile
from pathlib import Path
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

COS_IMAGE = "projects/cos-cloud/global/images/family/cos-stable"
_TOKEN = 'curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token | sed -E \'s/.*"access_token":"([^"]+)".*/\\1/\''


def _not_found(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 404 or type(exc).__name__ == "NotFound" or isinstance(exc, KeyError)


def exit_key(job_id: str, task_id: str, attempt: Any) -> str:
    return f"work/{job_id}/{task_id}/attempt-{attempt}.exit"


def startup_script(launch: LaunchSpec, project: str, zone: str, name: str, bucket: str, exit_object: str) -> str:
    """Bash for the VM's ``startup-script`` metadata (see the module docstring)."""
    env = dict(sorted(launch.env.items()))
    if launch.secret_env:
        env["WSR_SECRET_REFS"] = json.dumps(
            {var: f"gcpsm://{project}/{ref}" for var, ref in sorted(launch.secret_env.items())}
        )
    env_flags = " ".join(f"-e {shlex.quote(f'{k}={v}')}" for k, v in env.items())
    command = " ".join(shlex.quote(a) for a in launch.argv)
    instance_url = f"https://compute.googleapis.com/compute/v1/projects/{project}/zones/{zone}/instances/{name}"
    upload = f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media&name={exit_object}"
    return "\n".join(
        [
            "#!/bin/bash",
            "set -u",
            f"TOKEN=$({_TOKEN})",
            f"docker-credential-gcr configure-docker --registries={shlex.quote((launch.image or '').split('/')[0])} || true",
            f"docker run --rm --network host {env_flags} {shlex.quote(launch.image or '')} {command}",
            "CODE=$?",
            f'curl -s -X POST -H "Authorization: Bearer $TOKEN" --data-binary "$CODE" {shlex.quote(upload)}',
            f'curl -s -X DELETE -H "Authorization: Bearer $TOKEN" {shlex.quote(instance_url)}',
        ]
    )


class ComputeEngineBackend(ComputeBackend):
    """Self-deleting VMs in ``projects/<project>/zones/<zone>``.

    Args:
        project: Project id.
        zone: Zone (default ``<region>-b``).
        region: Region of the zone.
        status_store: :class:`~.storage.GcsObjectStore` the VMs write their exit codes to.
        client: An ``InstancesClient`` (tests inject a fake).
        log_store: Object store with the runtime's uploaded attempt logs (default: ``status_store``).
        service_account: Service account the VMs run as (needs storage, compute and Secret Manager access).
        disk_gb: Boot disk size.
        credentials: ProviderCredentials or google.auth credentials (default: ADC).
    """

    provider = Provider.GCP
    kind = ComputeKind.VM

    def __init__(
        self,
        project: str = "wsr-project",
        zone: Optional[str] = None,
        region: str = "europe-west1",
        status_store: Any = None,
        client: Any = None,
        log_store: Any = None,
        service_account: str = "default",
        disk_gb: int = 50,
        credentials: Any = None,
    ) -> None:
        self.project, self.region = project, region
        self.zone = zone or f"{region}-b"
        if status_store is None:
            raise ValueError("ComputeEngineBackend needs a GCS status_store for the VMs' exit codes")
        self.status_store = status_store
        self.log_store = log_store if log_store is not None else status_store
        self.service_account, self.disk_gb, self.credentials = service_account, disk_gb, credentials
        self._client = client
        self._spot: Dict[str, bool] = {}  # name -> spot, for polls after a self-delete
        self._cancelled: set = set()

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client("compute_instances", self.credentials)
        return self._client

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            provider=Provider.GCP,
            kinds=[ComputeKind.VM],
            max_wall_seconds=None,
            supports_spot=True,
            regions=[self.region],
            machine_types=dict(GCP_MACHINES),
        )

    def instance(self, launch: LaunchSpec) -> Dict[str, Any]:
        """The ``Instance`` resource for ``launch`` (a dict the client accepts)."""
        r = launch.resources
        name = resource_name(launch)
        exit_object = self.status_store.prefix + exit_key(launch.task.job_id, launch.task.task_id, launch.task.attempt)
        script = startup_script(launch, self.project, self.zone, name, self.status_store.bucket_name, exit_object)
        scheduling: Dict[str, Any] = {"automatic_restart": False, "on_host_maintenance": "TERMINATE"}
        if r.spot:
            scheduling.update(provisioning_model="SPOT", instance_termination_action="DELETE")
        if r.max_wall_seconds is not None:
            scheduling["max_run_duration"] = {"seconds": int(r.max_wall_seconds)}
            scheduling.setdefault("instance_termination_action", "DELETE")
        return {
            "name": name,
            "machine_type": f"zones/{self.zone}/machineTypes/{r.machine_type or DEFAULT_MACHINE}",
            "disks": [
                {
                    "boot": True,
                    "auto_delete": True,
                    "initialize_params": {"source_image": COS_IMAGE, "disk_size_gb": self.disk_gb},
                }
            ],
            "network_interfaces": [
                {"network": "global/networks/default", "access_configs": [{"name": "External NAT"}]}
            ],
            "service_accounts": [
                {"email": self.service_account, "scopes": ["https://www.googleapis.com/auth/cloud-platform"]}
            ],
            "scheduling": scheduling,
            "metadata": {"items": [{"key": "startup-script", "value": script}]},
            "labels": labels(launch),
        }

    def submit(self, launch: LaunchSpec) -> str:
        resource = self.instance(launch)
        try:
            self.client.insert(project=self.project, zone=self.zone, instance_resource=resource)
        except Exception as exc:
            raise_capacity(exc)
            raise
        self._spot[resource["name"]] = bool(launch.resources.spot)
        task = launch.task
        # The ids ride in the handle, so a poll after the VM deleted itself (or after a restart) finds its exit code.
        return f"gcp-vm:{self.project}/{self.zone}/{resource['name']}|{task.job_id}|{task.task_id}|{task.attempt}"

    @staticmethod
    def _parts(handle: str):
        location, job_id, task_id, attempt = handle.split(":", 1)[1].split("|")
        project, zone, name = location.split("/", 2)
        return project, zone, name, job_id, task_id, attempt

    def _exit_code(self, job_id: str, task_id: str, attempt: str) -> Optional[int]:
        key = exit_key(job_id, task_id, attempt)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "exit"
            try:
                self.status_store.get_file(key, target)
            except KeyError:
                return None
            text = target.read_text().strip()
        return int(text) if text.lstrip("-").isdigit() else 1

    def poll(self, handle: str) -> BackendStatus:
        if not handle.startswith("gcp-vm:") or handle.count("|") != 3:
            return BackendStatus(AttemptState.UNKNOWN, message="no such attempt")
        project, zone, name, job_id, task_id, attempt = self._parts(handle)
        try:
            vm = self.client.get(project=project, zone=zone, instance=name)
            status = state_name(field(vm, "status"))
            spot = state_name(field(vm, "scheduling.provisioning_model")) == "SPOT"
        except Exception as exc:
            if not _not_found(exc):
                return BackendStatus(AttemptState.UNKNOWN, message=str(exc))
            status, spot = "GONE", self._spot.get(name, False)
        if handle in self._cancelled:
            return BackendStatus(AttemptState.FAILED, exit_code=143, message="cancelled")
        if status in ("PROVISIONING", "STAGING"):
            return BackendStatus(AttemptState.PENDING, message=status.lower())
        code = self._exit_code(job_id, task_id, attempt)
        if code is not None:
            if code == 0:
                return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)
            return BackendStatus(AttemptState.FAILED, exit_code=code, message=f"task runtime exited with {code}")
        if status == "RUNNING":
            return BackendStatus(AttemptState.RUNNING)
        if spot:
            return BackendStatus(AttemptState.PREEMPTED, message="spot VM stopped before the task finished")
        return BackendStatus(AttemptState.FAILED, exit_code=1, message=f"VM {status.lower()} without an exit code")

    def cancel(self, handle: str) -> None:
        if handle in self._cancelled or self.poll(handle).state not in (AttemptState.PENDING, AttemptState.RUNNING):
            return
        project, zone, name, *_ = self._parts(handle)
        with contextlib.suppress(Exception):
            self.client.delete(project=project, zone=zone, instance=name)
        self._cancelled.add(handle)

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        if not handle.startswith("gcp-vm:") or handle.count("|") != 3:
            return ""
        project, zone, name, job_id, task_id, attempt = self._parts(handle)
        text = runtime_log(self.log_store, {"wsr_job_id": job_id, "wsr_task_id": task_id, "wsr_attempt": attempt}, tail)
        if text:
            return text
        with contextlib.suppress(Exception):
            output = str(
                field(self.client.get_serial_port_output(project=project, zone=zone, instance=name), "contents", "")
            )
            lines = output.splitlines()
            return "\n".join(lines if tail is None else lines[-tail:])
        return ""

    def price_per_hour(self, machine_type: Optional[str], region: Optional[str], spot: bool) -> Optional[float]:
        return price_per_hour(machine_type, region or self.region, spot)


__all__ = ["ComputeEngineBackend", "startup_script"]
