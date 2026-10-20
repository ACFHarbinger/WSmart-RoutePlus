"""Write-ahead Azure allocation intents and control-plane ownership reconciliation."""

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from logic.controllers.cloud.interfaces import CapacityError, ControlPlane, LaunchSpec
from logic.controllers.cloud.model import TaskState
from logic.controllers.cloud.providers.azure.orphan_queue import write_json


class AzureProvisioningMixin:
    """Keep allocation intent until the scheduler has durably accepted the handle.

    Reconciliation runs after lease expiration. It never deletes allocations of
    a currently placing owner or a task that holds the recorded backend handle.
    Each leased submission has unique resource names, fencing stale attempts.
    """

    _recovery_cp: Optional[ControlPlane] = None
    _recovery_scope: Optional[str] = None

    @property
    def provision_file(self) -> Path:
        return self.orphan_file.with_suffix(".provisions.json")

    def configure_recovery(self, control_plane: ControlPlane, scope: str) -> None:
        self._recovery_cp = control_plane
        self._recovery_scope = scope

    def _load_provisions(self) -> List[Dict[str, Any]]:
        try:
            with open(self.provision_file, encoding="utf-8") as source:
                records = json.load(source)
        except FileNotFoundError:
            return []
        required = ("token", "vm_name", "nic_name", "job_id", "task_id", "handle")
        if not isinstance(records, list) or any(
            not isinstance(record, dict) or any(not isinstance(record.get(key), str) for key in required)
            for record in records
        ):
            raise ValueError("Invalid Azure provisioning journal; preserve it for recovery")
        return records

    def _begin_provision(self, launch: LaunchSpec, vm_name: str, handle: str) -> str:
        token = uuid.uuid4().hex
        record = {
            "token": token,
            "scope": self._recovery_scope,
            "subscription_id": self.subscription_id,
            "profile": self.profile,
            "resource_group": self.resource_group,
            "vm_name": vm_name,
            "nic_name": f"nic-{vm_name}",
            "job_id": launch.task.job_id,
            "task_id": launch.task.task_id,
            "attempt": launch.task.attempt,
            "lease_owner": launch.task.lease_owner,
            "handle": handle,
        }
        with self._orphan_lock():
            records = self._load_provisions()
            records.append(record)
            write_json(self.provision_file, records)
        return token

    def _forget_provision(self, token: str) -> None:
        with self._orphan_lock():
            write_json(self.provision_file, [record for record in self._load_provisions() if record["token"] != token])

    def confirm_provision(self, handle: str) -> None:
        """Drop intents only after their exact handle is visible in durable task state."""
        self.reconcile_provisions()

    def _check_provision_lease(self, launch: LaunchSpec) -> None:
        if self._recovery_cp is None:
            return
        current = self._recovery_cp.get_task(launch.task.task_id)
        if (
            current.state != TaskState.PLACING
            or current.attempt != launch.task.attempt
            or current.lease_owner != launch.task.lease_owner
        ):
            raise CapacityError("Azure provisioning lease was lost; abandon this allocation")

    def reconcile_provisions(self) -> None:
        """Reconcile journal records against authoritative task/attempt ownership.

        Unknown tasks/control planes are retained for investigation. Do not use
        lease timestamps here: the control plane must first atomically expire
        placing leases, fencing an old owner before cleanup can begin.
        """
        if self._recovery_cp is None:
            return
        with self._orphan_lock():
            records = self._load_provisions()
        for record in records:
            if (
                record.get("scope") != self._recovery_scope
                or record.get("subscription_id") != self.subscription_id
                or record.get("profile") != self.profile
            ):
                continue
            try:
                task = self._recovery_cp.get_task(record["task_id"])
            except KeyError:
                continue  # Missing ownership is not permission to delete an unknown workload.
            if task.job_id != record["job_id"]:
                raise ValueError("Azure intent does not match control-plane job ownership")
            if task.backend_handle == record["handle"] and task.attempt == record["attempt"]:
                self._forget_provision(record["token"])
                continue
            if (
                task.state == TaskState.PLACING
                and task.attempt == record["attempt"]
                and task.lease_owner == record["lease_owner"]
            ):
                continue
            # Unique names ensure this intent cannot address a newer owner's VM.
            self._teardown_failed_submit(record["resource_group"], record["vm_name"], record["nic_name"])
            # Failed deletions have been durably transferred into the cleanup queue.
            self._forget_provision(record["token"])
