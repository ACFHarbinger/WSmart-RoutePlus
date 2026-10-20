"""In-process fakes of the Google Cloud clients the GCP provider uses (CR-05).

Each fake keeps state in memory, accepts the dict requests the backends build, and
answers with dicts (the backends read both dicts and SDK messages). Attempt outcomes
come from ``WSR_FAKE_OUTCOME`` in the request's environment (``testing.launches``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from logic.controllers.cloud.testing import outcome_of


class NotFound(Exception):
    code = 404


class AlreadyExists(Exception):
    code = 409


class ResourceExhausted(Exception):
    code = 429


# --------------------------------------------------------------------------- storage


class _Blob:
    def __init__(self, store: Dict[str, bytes], name: str) -> None:
        self.store, self.name = store, name

    def upload_from_filename(self, path: str) -> None:
        self.store[self.name] = Path(path).read_bytes()

    def download_to_filename(self, path: str) -> None:
        if self.name not in self.store:
            raise NotFound(self.name)
        Path(path).write_bytes(self.store[self.name])

    def exists(self) -> bool:
        return self.name in self.store

    def delete(self) -> None:
        if self.store.pop(self.name, None) is None:
            raise NotFound(self.name)


class _Bucket:
    def __init__(self, store: Dict[str, bytes]) -> None:
        self.store = store

    def blob(self, name: str) -> _Blob:
        return _Blob(self.store, name)


class FakeGcsClient:
    def __init__(self) -> None:
        self.buckets: Dict[str, Dict[str, bytes]] = {}

    def bucket(self, name: str) -> _Bucket:
        return _Bucket(self.buckets.setdefault(name, {}))

    def list_blobs(self, bucket: str, prefix: str = "") -> List[Any]:
        return [type("Blob", (), {"name": n})() for n in sorted(self.buckets.get(bucket, {})) if n.startswith(prefix)]


# --------------------------------------------------------------------------- secrets


class FakeSecretManagerClient:
    def __init__(self) -> None:
        self.secrets: Dict[str, List[bytes]] = {}

    def create_secret(self, parent: str, secret_id: str, secret: Dict[str, Any]) -> Dict[str, Any]:
        name = f"{parent}/secrets/{secret_id}"
        if name in self.secrets:
            raise AlreadyExists(name)
        self.secrets[name] = []
        return {"name": name}

    def add_secret_version(self, parent: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if parent not in self.secrets:
            raise NotFound(parent)
        self.secrets[parent].append(payload["data"])
        return {"name": f"{parent}/versions/{len(self.secrets[parent])}"}

    def access_secret_version(self, name: str) -> Dict[str, Any]:
        secret = name.rsplit("/versions/", 1)[0]
        if not self.secrets.get(secret):
            raise NotFound(name)
        return {"payload": {"data": self.secrets[secret][-1]}}

    def delete_secret(self, name: str) -> None:
        if self.secrets.pop(name, None) is None:
            raise NotFound(name)


# --------------------------------------------------------------------------- Batch


class FakeBatchClient:
    """Jobs move QUEUED -> RUNNING -> outcome over successive ``get_job`` calls."""

    def __init__(self, capacity: Optional[int] = None) -> None:
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.capacity = capacity

    def create_job(self, parent: str, job: Dict[str, Any], job_id: str) -> Dict[str, Any]:
        live = [j for j in self.jobs.values() if j["status"]["state"] in ("QUEUED", "RUNNING")]
        if self.capacity is not None and len(live) >= self.capacity:
            raise ResourceExhausted("Quota 'CPUS' exceeded")
        name = f"{parent}/jobs/{job_id}"
        env = job["task_groups"][0]["task_spec"]["environment"]["variables"]
        self.jobs[name] = {
            "spec": job,
            "outcome": outcome_of(env),
            "polls": 0,
            "labels": job["labels"],
            "status": {"state": "QUEUED", "status_events": []},
        }
        return {"name": name}

    def get_job(self, name: str) -> Dict[str, Any]:
        if name not in self.jobs:
            raise NotFound(name)
        job = self.jobs[name]
        job["polls"] += 1
        status = job["status"]
        if job["polls"] >= 2 and status["state"] == "QUEUED":
            status["state"] = "RUNNING"
        elif job["polls"] >= 3 and status["state"] == "RUNNING" and job["outcome"] != "hang":
            if job["outcome"] == "succeed":
                status["state"] = "SUCCEEDED"
            else:
                code = 50001 if job["outcome"] == "preempt" else 3
                status["state"] = "FAILED"
                status["status_events"].append(
                    {"description": f"Task state is updated from RUNNING to FAILED with exit code {code}."}
                )
        return {"name": name, "status": status, "labels": job["labels"]}

    def delete_job(self, name: str) -> None:
        if self.jobs.pop(name, None) is None:
            raise NotFound(name)


# --------------------------------------------------------------------------- Cloud Run


class _Operation:
    def __init__(self, metadata_name: Optional[str] = None, result: Any = None) -> None:
        self.metadata = type("Meta", (), {"name": metadata_name})() if metadata_name else None
        self._result = result

    def result(self) -> Any:
        return self._result


class FakeRunClients:
    """``jobs`` and ``executions`` clients sharing one state."""

    def __init__(self) -> None:
        self.jobs_by_name: Dict[str, Dict[str, Any]] = {}
        self.executions_by_name: Dict[str, Dict[str, Any]] = {}
        self.deleted: List[str] = []
        outer = self

        class Jobs:
            def create_job(self, parent: str, job: Dict[str, Any], job_id: str) -> _Operation:
                name = f"{parent}/jobs/{job_id}"
                outer.jobs_by_name[name] = job
                return _Operation(result={"name": name})

            def run_job(self, name: str) -> _Operation:
                job = outer.jobs_by_name[name]
                container = job["template"]["template"]["containers"][0]
                execution = f"{name}/executions/{name.rsplit('/', 1)[-1]}-x1"
                outer.executions_by_name[execution] = {
                    "outcome": outcome_of(container["env"]),
                    "polls": 0,
                    "cancelled": False,
                    "labels": job["labels"],
                }
                return _Operation(metadata_name=execution)

            def delete_job(self, name: str) -> None:
                outer.deleted.append(name)
                outer.jobs_by_name.pop(name, None)

        class Executions:
            def get_execution(self, name: str) -> Dict[str, Any]:
                if name not in outer.executions_by_name:
                    raise NotFound(name)
                e = outer.executions_by_name[name]
                e["polls"] += 1
                counts = {"running_count": 0, "succeeded_count": 0, "failed_count": 0, "cancelled_count": 0}
                if e["cancelled"]:
                    counts["cancelled_count"] = 1
                elif e["polls"] >= 2 and (e["outcome"] == "hang" or e["polls"] < 3):
                    counts["running_count"] = 1
                elif e["polls"] >= 3:
                    counts["succeeded_count" if e["outcome"] == "succeed" else "failed_count"] = 1
                conditions = [{"message": "Task failed with exit code 3"}] if counts["failed_count"] else []
                return {"name": name, "labels": e["labels"], "conditions": conditions, **counts}

            def cancel_execution(self, name: str) -> None:
                outer.executions_by_name[name]["cancelled"] = True

        self.jobs, self.executions = Jobs(), Executions()


# --------------------------------------------------------------------------- Compute Engine


class FakeInstancesClient:
    """VMs that 'run' their startup script: PROVISIONING -> RUNNING -> write the exit object and delete themselves."""

    def __init__(self, gcs: FakeGcsClient) -> None:
        self.gcs = gcs
        self.vms: Dict[str, Dict[str, Any]] = {}
        self.inserted: List[Dict[str, Any]] = []

    def insert(self, project: str, zone: str, instance_resource: Dict[str, Any]) -> _Operation:
        script = instance_resource["metadata"]["items"][0]["value"]
        match = re.search(r"WSR_FAKE_OUTCOME=(\w+)", script)
        upload = re.search(r"/b/([^/]+)/o\?uploadType=media&name=(\S+?)'?$", script, re.M)
        self.inserted.append(instance_resource)
        self.vms[instance_resource["name"]] = {
            "resource": instance_resource,
            "outcome": match.group(1) if match else "succeed",
            "bucket": upload.group(1),
            "exit_object": upload.group(2),
            "polls": 0,
            "status": "PROVISIONING",
        }
        return _Operation()

    def get(self, project: str, zone: str, instance: str) -> Dict[str, Any]:
        vm = self.vms.get(instance)
        if vm is None:
            raise NotFound(instance)
        vm["polls"] += 1
        if vm["polls"] >= 2 and vm["status"] == "PROVISIONING":
            vm["status"] = "RUNNING"
        elif vm["polls"] >= 3 and vm["status"] == "RUNNING" and vm["outcome"] != "hang":
            if vm["outcome"] != "preempt":  # a preempted spot VM is deleted without writing its exit code
                code = "0" if vm["outcome"] == "succeed" else "3"
                self.gcs.buckets.setdefault(vm["bucket"], {})[vm["exit_object"]] = code.encode()
            del self.vms[instance]  # self-delete
            raise NotFound(instance)
        return {
            "name": instance,
            "status": vm["status"],
            "labels": vm["resource"]["labels"],
            "scheduling": vm["resource"]["scheduling"],
        }

    def delete(self, project: str, zone: str, instance: str) -> _Operation:
        if self.vms.pop(instance, None) is None:
            raise NotFound(instance)
        return _Operation()

    def get_serial_port_output(self, project: str, zone: str, instance: str) -> Dict[str, Any]:
        return {"contents": f"serial output of {instance}\nline 2"}
