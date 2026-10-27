"""Scoped, locked persistence for Azure VM cleanup retries.

Unknown account records are retained, never acted on with another account's client.
This queue tracks failed cleanup; it is not a write-ahead provisioning journal.
"""

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Protocol, cast


def write_json(path: Path, records: List[Dict[str, Any]]) -> None:
    """Persist a journal update before returning (including the directory entry)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            delete=False,
        ) as target:
            temporary = Path(target.name)
            json.dump(records, target, indent=2)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _AzureVmCleanup(Protocol):
    """Deletion helpers supplied by AzureVmBackend. Not a runtime base."""

    def _delete_vm_resource(self, rg: str, vm_name: str) -> None: ...

    def _cleanup_nic_in_rg(self, rg: str, nic_name: str) -> None: ...

    def _is_not_found(self, exc: Exception) -> bool: ...


class AzureVmOrphanQueueMixin:
    """Persistent cleanup queue shared by the VM backend's submit and poll paths."""

    orphan_file: Path
    subscription_id: str
    profile: str
    resource_group: str

    @contextmanager
    def _orphan_lock(self) -> Generator[None, None, None]:
        import fcntl

        self.orphan_file.parent.mkdir(parents=True, exist_ok=True)
        with open(str(self.orphan_file) + ".lock", "a", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _load_orphans(self) -> List[Dict[str, Any]]:
        try:
            with open(self.orphan_file, encoding="utf-8") as source:
                records = json.load(source)
        except FileNotFoundError:
            return []
        if not isinstance(records, list) or any(
            not isinstance(record, dict)
            or record.get("kind") not in ("vm", "nic")
            or not isinstance(record.get("name"), str)
            or not record["name"]
            for record in records
        ):
            raise ValueError("Invalid Azure orphan queue; preserve it for recovery")
        return records

    def _save_orphans(self, orphans: List[Dict[str, Any]]) -> None:
        write_json(self.orphan_file, orphans)

    def _record_orphan(self, kind: str, resource_group: str, name: str, error: Optional[str] = None) -> None:
        """Record an uncleaned resource into the persistent orphan queue."""
        with self._orphan_lock():
            orphans = self._load_orphans()
            now = time.time()
            for record in orphans:
                if (
                    record.get("subscription_id") == self.subscription_id
                    and record.get("profile") == self.profile
                    and record.get("kind") == kind
                    and record.get("resource_group") == resource_group
                    and record.get("name") == name
                ):
                    record["attempts"] = record.get("attempts", 0) + 1
                    record["last_attempt"] = now
                    if error:
                        record["last_error"] = str(error)
                    self._save_orphans(orphans)
                    return
            orphans.append(
                {
                    "subscription_id": self.subscription_id,
                    "profile": self.profile,
                    "kind": kind,
                    "resource_group": resource_group,
                    "name": name,
                    "recorded_at": now,
                    "last_attempt": now,
                    "attempts": 1,
                    "last_error": str(error) if error else None,
                }
            )
            self._save_orphans(orphans)

    def drain_orphans(self) -> List[str]:
        """Retry deletion for all recorded orphan resources; returns list of cleaned resource identifiers."""
        with self._orphan_lock():
            orphans = self._load_orphans()
            if not orphans:
                return []

            remaining: List[Dict[str, Any]] = []
            cleaned: List[str] = []

            # Sort so that VMs are deleted before NICs (NIC cannot be deleted while attached to VM)
            orphans.sort(key=lambda r: 0 if r.get("kind") == "vm" else 1)

            for record in orphans:
                if record.get("subscription_id") != self.subscription_id or record.get("profile") != self.profile:
                    remaining.append(record)
                    continue
                kind = record.get("kind")
                rg = record.get("resource_group", self.resource_group)
                name = record.get("name", "")
                if not name:
                    continue

                success = False
                cleanup = cast(_AzureVmCleanup, self)
                try:
                    if kind == "vm":
                        cleanup._delete_vm_resource(rg, name)
                        success = True
                    elif kind == "nic":
                        cleanup._cleanup_nic_in_rg(rg, name)
                        success = True
                except Exception as exc:
                    if cleanup._is_not_found(exc):
                        success = True
                    else:
                        record["attempts"] = record.get("attempts", 0) + 1
                        record["last_attempt"] = time.time()
                        record["last_error"] = str(exc)
                        remaining.append(record)

                if success:
                    cleaned.append(f"{kind}:{rg}:{name}")

            self._save_orphans(remaining)
            return cleaned
