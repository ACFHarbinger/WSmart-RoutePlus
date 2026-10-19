"""Runner behind the ``cloud`` command of ``main.py``.

Attributes:
    local_service: The local profile: SQLite control plane, filesystem store, subprocess backend.
    run_cloud: Execute one ``cloud`` sub-command.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

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


def _secrets(opts: Dict[str, Any]) -> Any:
    """``secrets set|delete|exists NAME``: the value is read from a file or stdin and never printed."""
    import sys

    from logic.controllers.cloud.auth.credentials import KeychainSecretStore, secret_account

    name = opts["name"]
    account = name if "/" in name else secret_account(name)
    store = KeychainSecretStore()
    if opts["action"] == "set":
        if opts.get("from_file"):
            value = Path(opts["from_file"]).read_text()
        elif opts.get("stdin"):
            value = sys.stdin.read()
        else:
            raise ValueError("secrets set needs --from-file or --stdin (values are never taken as arguments)")
        if not value.strip():
            raise ValueError("refusing to store an empty secret")
        store.put(account, value)
        _print(opts, {"name": name, "stored": True}, f"stored {name}")
        return True
    if opts["action"] == "delete":
        store.delete(account)
        _print(opts, {"name": name, "deleted": True}, f"deleted {name}")
        return True
    exists = store.get(account) is not None
    _print(opts, {"name": name, "exists": exists}, f"{name}: {'present' if exists else 'absent'}")
    return exists


@dataclass
class _Context:
    """What every sub-command handler gets."""

    opts: Dict[str, Any]
    root: Path
    project_root: Path
    service: CloudService


def _submit(ctx: _Context) -> Any:
    opts, service = ctx.opts, ctx.service
    overrides = _spec_overrides(opts)
    if opts.get("manifest"):
        spec = load_manifest(Path(opts["manifest"]), policy=opts.get("policy"), **overrides)
    else:
        spec = job_from_command(opts["sim_command"], **overrides)
    job = service.submit(spec)
    payload: Dict[str, Any] = {"job_id": job.job_id, "tasks": len(spec.cells)}
    if not opts.get("wait") and not opts.get("no_worker"):
        payload["worker_pid"] = spawn_worker(ctx.root, project_root=ctx.project_root, python=service.python)
    _print(opts, payload, f"submitted {job.job_id} ({len(spec.cells)} task(s))")
    if opts.get("wait"):
        state = service.run(job.job_id, poll_seconds=float(opts.get("poll") or 5.0))
        payload["state"] = state.value if state else None
        _print(opts, payload, f"{job.job_id}: {payload['state']}")
    return payload


def _status(ctx: _Context) -> Any:
    opts, service = ctx.opts, ctx.service
    if opts.get("job_id"):
        status = service.status(opts["job_id"])
        lines = [f"{status['job_id']} {status['name']}: {status['state']} {status['counts']}"]
        lines += [f"  {t['task_id']} {t['cell_id']}: {t['state']} (attempt {t['attempt']})" for t in status["tasks"]]
        _print(opts, status, "\n".join(lines))
        return status
    jobs = [service.status(job.job_id) for job in service.cp.list_jobs()]
    summary = [{key: job[key] for key in ("job_id", "name", "state", "counts")} for job in jobs]
    _print(opts, summary, "\n".join(f"{j['job_id']} {j['name']}: {j['state']} {j['counts']}" for j in summary))
    return summary


def _logs(ctx: _Context) -> Any:
    text = ctx.service.logs(ctx.opts["task_id"], ctx.opts.get("tail"))
    _print(ctx.opts, {"task_id": ctx.opts["task_id"], "log": text}, text)
    return text


def _cancel(ctx: _Context) -> Any:
    job_id = ctx.opts["job_id"]
    ctx.service.cancel(job_id)
    _print(ctx.opts, {"job_id": job_id, "cancel_requested": True}, f"cancel requested for {job_id}")
    return None


def _retry(ctx: _Context) -> Any:
    count = ctx.service.retry(ctx.opts["job_id"])
    _print(ctx.opts, {"job_id": ctx.opts["job_id"], "retried": count}, f"requeued {count} task(s)")
    return count


def _download(ctx: _Context) -> Any:
    job_id = ctx.opts["job_id"]
    dest = Path(ctx.opts.get("dest") or ctx.root / "results" / job_id)
    files = ctx.service.download(job_id, dest)
    listing = [
        {"relpath": path.relative_to(dest).as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size}
        for path in files
    ]
    payload = {"job_id": job_id, "dest": str(dest), "files": len(files), "file_list": listing}
    _print(ctx.opts, payload, f"{len(files)} file(s) in {dest}")
    return files


def _backends(ctx: _Context) -> Any:
    backends = [
        {"provider": provider.value, "compute": kind.value, "capabilities": to_dict(backend.capabilities())}
        for (provider, kind), backend in ctx.service.backends.items()
    ]
    _print(ctx.opts, backends, "\n".join(f"{b['provider']}/{b['compute']}" for b in backends))
    return backends


def _credentials(ctx: _Context) -> Any:
    from logic.controllers.cloud.auth.credentials import detect_logins

    logins = [to_dict(status) for status in detect_logins()]
    _print(ctx.opts, logins, "\n".join(f"{s['provider']}: {'usable' if s['usable'] else 'no login'}" for s in logins))
    return logins


def _profiles(ctx: _Context) -> Any:
    from logic.controllers.cloud.auth.config import load_profiles

    profiles = [to_dict(profile) for profile in load_profiles().values()]
    _print(ctx.opts, profiles, "\n".join(f"{p['name']}: {p['provider']}/{p['compute']}" for p in profiles))
    return profiles


def _worker(ctx: _Context) -> Any:
    opts, service = ctx.opts, ctx.service
    poll = float(opts.get("poll") or 5.0)
    if opts.get("once"):
        changes = service.tick()
        _print(opts, {"changes": changes}, f"{changes} change(s)")
        return changes
    if opts.get("job"):
        state = service.run(opts["job"], poll_seconds=poll)
        _print(opts, {"state": state.value if state else None}, f"{opts['job']}: {state.value if state else None}")
        return state
    reason = serve(service, ctx.root, poll_seconds=poll)
    _print(opts, {"stopped": reason}, reason)
    return reason


_COMMANDS: Dict[str, Callable[[_Context], Any]] = {
    "submit": _submit,
    "status": _status,
    "logs": _logs,
    "cancel": _cancel,
    "retry": _retry,
    "download": _download,
    "backends": _backends,
    "credentials": _credentials,
    "profiles": _profiles,
    "secrets": lambda ctx: _secrets(ctx.opts),
    "worker": _worker,
}


def run_cloud(opts: Dict[str, Any]) -> Any:
    """Execute one ``cloud`` sub-command; returns its payload."""
    command = opts["cloud_command"]
    if command not in _COMMANDS:
        raise ValueError(f"unknown cloud command {command!r}")
    root = Path(opts.get("root") or "outputs/cloud")
    project_root = Path(opts["project_root"]) if opts.get("project_root") else REPO_ROOT
    return _COMMANDS[command](_Context(opts, root, project_root, local_service(root, project_root)))
