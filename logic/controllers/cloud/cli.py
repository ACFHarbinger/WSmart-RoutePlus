"""Runner behind the ``cloud`` command of ``main.py``.

Attributes:
    local_service: The local profile: SQLite control plane, filesystem store, subprocess backend.
    run_cloud: Execute one ``cloud`` sub-command.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from logic.controllers.cloud.local.backend import LocalProcessBackend
from logic.controllers.cloud.local.control_plane import SQLiteControlPlane
from logic.controllers.cloud.local.object_store import EnvSecretStore, FilesystemObjectStore
from logic.controllers.cloud.manifest import job_from_command, load_manifest
from logic.controllers.cloud.model import (
    ComputeKind,
    GurobiMode,
    Provider,
    Resources,
    to_dict,
)
from logic.controllers.cloud.service import CloudService
from logic.controllers.cloud.storage import sha256_file
from logic.controllers.cloud.worker import serve, spawn_worker

REPO_ROOT = Path(__file__).resolve().parents[3]


def local_service(root: Path, project_root: Optional[Path] = None) -> CloudService:
    """The local profile rooted at ``root``; local tasks run ``main.py`` in ``project_root``.

    Secrets come from ``WSR_SECRET_<NAME>`` environment variables.
    """
    root = Path(root)
    project_root = Path(project_root) if project_root else REPO_ROOT
    secrets = EnvSecretStore(prefix="WSR_SECRET_")
    return CloudService(
        control_plane=SQLiteControlPlane(root / "control.db"),
        store=FilesystemObjectStore(root / "objects"),
        backends={(Provider.LOCAL, ComputeKind.PROCESS): LocalProcessBackend(cwd=project_root, secrets=secrets)},
        workroot=root / "work",
    )


def _spec_overrides(opts: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {"name": opts.get("name"), "max_parallel": opts.get("max_parallel")}
    if opts.get("provider"):
        overrides["provider"] = Provider(opts["provider"])
    if opts.get("compute"):
        overrides["compute"] = ComputeKind(opts["compute"])
    if opts.get("gurobi"):
        overrides["gurobi"] = GurobiMode(opts["gurobi"])
    overrides["gurobi_secret"] = opts.get("gurobi_secret")
    resource_fields = {
        "cpus": opts.get("cpus"),
        "memory_gb": opts.get("memory_gb"),
        "max_retries": opts.get("retries"),
        "max_wall_seconds": opts.get("max_wall"),
        "idle_seconds": opts.get("idle"),
        "spot": True if opts.get("spot") else None,
    }
    if any(value is not None for value in resource_fields.values()):
        defaults = to_dict(Resources())
        defaults.update({key: value for key, value in resource_fields.items() if value is not None})
        overrides["resources"] = Resources(**defaults)
    return overrides


def _print(opts: Dict[str, Any], payload: Any, text: str) -> None:
    print(json.dumps(payload, indent=1) if opts.get("json") else text)


def run_cloud(opts: Dict[str, Any]) -> Any:
    """Execute one ``cloud`` sub-command; returns its payload."""
    root = Path(opts.get("root") or "outputs/cloud")
    project_root = Path(opts["project_root"]) if opts.get("project_root") else REPO_ROOT
    service = local_service(root, project_root)
    command = opts["cloud_command"]

    if command == "submit":
        overrides = _spec_overrides(opts)
        if opts.get("manifest"):
            spec = load_manifest(Path(opts["manifest"]), policy=opts.get("policy"), **overrides)
        else:
            spec = job_from_command(opts["sim_command"], **overrides)
        job = service.submit(spec)
        payload: Dict[str, Any] = {"job_id": job.job_id, "tasks": len(spec.cells)}
        if not opts.get("wait") and not opts.get("no_worker"):
            payload["worker_pid"] = spawn_worker(root, project_root=project_root, python=service.python)
        _print(opts, payload, f"submitted {job.job_id} ({len(spec.cells)} task(s))")
        if opts.get("wait"):
            state = service.run(job.job_id, poll_seconds=float(opts.get("poll") or 5.0))
            payload["state"] = state.value if state else None
            _print(opts, payload, f"{job.job_id}: {payload['state']}")
        return payload

    if command == "status":
        if opts.get("job_id"):
            status = service.status(opts["job_id"])
            lines = [f"{status['job_id']} {status['name']}: {status['state']} {status['counts']}"]
            lines += [
                f"  {t['task_id']} {t['cell_id']}: {t['state']} (attempt {t['attempt']})" for t in status["tasks"]
            ]
            _print(opts, status, "\n".join(lines))
            return status
        jobs = [service.status(job.job_id) for job in service.cp.list_jobs()]
        summary = [{key: job[key] for key in ("job_id", "name", "state", "counts")} for job in jobs]
        _print(opts, summary, "\n".join(f"{j['job_id']} {j['name']}: {j['state']} {j['counts']}" for j in summary))
        return summary

    if command == "logs":
        text = service.logs(opts["task_id"], opts.get("tail"))
        _print(opts, {"task_id": opts["task_id"], "log": text}, text)
        return text

    if command == "cancel":
        service.cancel(opts["job_id"])
        _print(opts, {"job_id": opts["job_id"], "cancel_requested": True}, f"cancel requested for {opts['job_id']}")
        return None

    if command == "retry":
        count = service.retry(opts["job_id"])
        _print(opts, {"job_id": opts["job_id"], "retried": count}, f"requeued {count} task(s)")
        return count

    if command == "download":
        dest = Path(opts.get("dest") or root / "results" / opts["job_id"])
        files = service.download(opts["job_id"], dest)
        listing = [
            {"relpath": path.relative_to(dest).as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size}
            for path in files
        ]
        payload = {"job_id": opts["job_id"], "dest": str(dest), "files": len(files), "file_list": listing}
        _print(opts, payload, f"{len(files)} file(s) in {dest}")
        return files

    if command == "backends":
        backends = [
            {"provider": provider.value, "compute": kind.value, "capabilities": to_dict(backend.capabilities())}
            for (provider, kind), backend in service.backends.items()
        ]
        _print(opts, backends, "\n".join(f"{b['provider']}/{b['compute']}" for b in backends))
        return backends

    if command == "worker":
        if opts.get("once"):
            changes = service.tick()
            _print(opts, {"changes": changes}, f"{changes} change(s)")
            return changes
        if opts.get("job"):
            state = service.run(opts["job"], poll_seconds=float(opts.get("poll") or 5.0))
            _print(opts, {"state": state.value if state else None}, f"{opts['job']}: {state.value if state else None}")
            return state
        reason = serve(service, root, poll_seconds=float(opts.get("poll") or 5.0))
        _print(opts, {"stopped": reason}, reason)
        return reason

    raise ValueError(f"unknown cloud command {command!r}")
