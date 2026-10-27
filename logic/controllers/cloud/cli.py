"""Runner behind the ``cloud`` command of ``main.py``.

Attributes:
    local_service: The local profile: SQLite control plane, filesystem store, subprocess backend.
    run_cloud: Execute one ``cloud`` sub-command.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

from logic.controllers.cloud.local.backend import LocalProcessBackend
from logic.controllers.cloud.local.control_plane import SQLiteControlPlane
from logic.controllers.cloud.local.object_store import EnvSecretStore, FilesystemObjectStore
from logic.controllers.cloud.manifest import job_from_command, load_manifest
from logic.controllers.cloud.model import (
    ComputeKind,
    GurobiMode,
    JobState,
    PlacementTarget,
    Provider,
    Resources,
    to_dict,
)
from logic.controllers.cloud.service import CloudService
from logic.controllers.cloud.store.storage import sha256_file
from logic.controllers.cloud.worker import serve, spawn_worker, worker_lock

REPO_ROOT = Path(__file__).resolve().parents[3]


def pool_worker_service(root: Path, project_root: Path, slots: int) -> CloudService:
    """A warm pool worker (CR-09): the shared control plane and store (``WSR_CLOUD_CONTROL_PLANE`` and
    ``WSR_CLOUD_STORE`` on a cloud machine, else the local profile) and this machine's processes."""
    import os

    from logic.controllers.cloud.profiles import control_plane_from_uri, store_from_uri
    from logic.controllers.cloud.service import RemoteConfig

    cp_uri, store_uri = os.environ.get("WSR_CLOUD_CONTROL_PLANE"), os.environ.get("WSR_CLOUD_STORE")
    root = Path(root)
    cp = control_plane_from_uri(cp_uri) if cp_uri else SQLiteControlPlane(root / "control.db")
    store = store_from_uri(store_uri) if store_uri else FilesystemObjectStore(root / "objects")
    remote = (
        RemoteConfig(control_plane_uri=cp_uri, store_uri=store_uri, project_root=project_root)
        if cp_uri and store_uri
        else None
    )
    secrets = EnvSecretStore(prefix="WSR_SECRET_")
    return CloudService(
        control_plane=cp,
        store=store,
        backends={(Provider.LOCAL, ComputeKind.PROCESS): LocalProcessBackend(cwd=project_root, secrets=secrets)},
        workroot=root / "work",
        remote=remote,
        pool_mode=True,
        pool_slots=slots,
    )


def local_service(root: Path, project_root: Optional[Path] = None) -> CloudService:
    """The local profile rooted at ``root``; local tasks run ``main.py`` in ``project_root``.

    Secrets come from ``WSR_SECRET_<NAME>`` environment variables.
    """
    root = Path(root)
    project_root = Path(project_root) if project_root else REPO_ROOT
    secrets = EnvSecretStore(prefix="WSR_SECRET_")
    cp = SQLiteControlPlane(root / "control.db")
    from logic.controllers.cloud.cost import BudgetPolicy, PriceTable

    budget_file = root / "budget.json"
    budget = BudgetPolicy(**json.loads(budget_file.read_text())) if budget_file.is_file() else None
    pool_file = root / "pool.json"  # CR-09 warm pool: {"target": "aws/vm/-/c6i.xlarge", "cpus": 4, ...}
    pool = None
    if pool_file.is_file():
        from logic.controllers.cloud.scheduler import PoolPolicy

        data = json.loads(pool_file.read_text())
        fields = cast(Dict[str, Any], {**data, "target": parse_target(data["target"])})
        pool = PoolPolicy(**fields)

    def predictor():
        from logic.controllers.cloud.prediction import fitted_predictor

        return fitted_predictor(cp)

    return CloudService(
        control_plane=cp,
        store=FilesystemObjectStore(root / "objects"),
        backends={(Provider.LOCAL, ComputeKind.PROCESS): LocalProcessBackend(cwd=project_root, secrets=secrets)},
        workroot=root / "work",
        predictor_factory=predictor,
        prices=PriceTable(root / "prices.json"),
        budget=budget,
        pool=pool,
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
    if opts.get("priority") is not None:
        overrides["priority"] = int(opts["priority"])
    if opts.get("target"):
        overrides["placement"] = [parse_target(t) for t in opts["target"]]
    if opts.get("pool"):
        overrides["pool"] = True
    resource_fields = {
        "cpus": opts.get("cpus"),
        "memory_gb": opts.get("memory_gb"),
        "max_retries": opts.get("retries"),
        "max_wall_seconds": opts.get("max_wall"),
        "idle_seconds": opts.get("idle"),
        "spot": True if opts.get("spot") else None,
        "machine_type": opts.get("machine_type"),
        "region": opts.get("region"),
        "budget_usd": opts.get("budget"),
    }
    if any(value is not None for value in resource_fields.values()):
        defaults = to_dict(Resources())
        defaults.update({key: value for key, value in resource_fields.items() if value is not None})
        overrides["resources"] = Resources(**defaults)
    return overrides


def parse_target(text: str) -> PlacementTarget:
    """``provider/compute[/region[/machine_type[/spot]]][:max_tasks]`` (``-`` skips a field)."""
    body, _, quota = text.partition(":")
    parts = body.split("/")
    if len(parts) < 2:
        raise ValueError(f"placement target {text!r}: expected provider/compute[/region[/machine_type[/spot]]]")
    region = parts[2] if len(parts) > 2 and parts[2] != "-" else None
    machine = parts[3] if len(parts) > 3 and parts[3] != "-" else None
    spot = len(parts) > 4 and parts[4] == "spot"
    return PlacementTarget(
        Provider(parts[0]), ComputeKind(parts[1]), region, machine, spot, int(quota) if quota else None
    )


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
    profile = opts.get("_profile")
    if profile is not None:  # the profile's provider, compute kind, region, image and license by default
        for key, value in (("provider", profile.provider), ("compute", profile.compute), ("region", profile.region)):
            opts[key] = opts.get(key) or value
        opts["gurobi_secret"] = opts.get("gurobi_secret") or profile.gurobi_secret
    overrides = _spec_overrides(opts)
    if profile is not None and profile.image and not overrides.get("image"):
        overrides["image"] = profile.image
    if opts.get("manifest"):
        spec = load_manifest(Path(opts["manifest"]), policy=opts.get("policy"), **overrides)
    else:
        spec = job_from_command(opts["sim_command"], **overrides)
    job = service.submit(spec)
    payload: Dict[str, Any] = {"job_id": job.job_id, "tasks": len(spec.cells)}
    if not opts.get("wait") and not opts.get("no_worker"):
        payload["worker_pid"] = spawn_worker(
            ctx.root, project_root=ctx.project_root, python=service.python, profile=opts.get("profile")
        )
    _print(opts, payload, f"submitted {job.job_id} ({len(spec.cells)} task(s))")
    if opts.get("wait"):
        import time

        poll_sec = float(opts.get("poll") or 5.0)
        with worker_lock(ctx.root) as mine:
            if mine:
                state = service.run(job.job_id, poll_seconds=poll_sec)
            else:
                state = None
                while True:
                    st = service.status(job.job_id)
                    job_st = JobState(st["state"])
                    if job_st in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
                        state = job_st
                        break
                    time.sleep(poll_sec)
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


def _metrics(ctx: _Context) -> Any:
    payload = ctx.service.metrics(ctx.opts["task_id"], since=ctx.opts.get("since"))
    progress = payload["progress"]
    text = f"{payload['task_id']}: {len(payload['points'])} resource sample(s)" + (
        f", day {progress['day']}/{progress['days'] or '?'} ({progress['policy']})" if progress else ""
    )
    _print(ctx.opts, payload, text)
    return payload


def _track(ctx: _Context) -> Any:
    from logic.controllers.cloud.telemetry import import_job
    from logic.src.constants import ROOT_DIR
    from logic.src.tracking import _DEFAULT_TRACKING_URI
    from logic.src.tracking.core.store import TrackingStore

    # The same database the simulator's tracker writes (tracking.db under the tracking directory).
    uri = Path(ctx.opts.get("tracking_uri") or Path(ROOT_DIR) / _DEFAULT_TRACKING_URI)
    store = TrackingStore(str(uri / "tracking.db"))
    job_id = ctx.opts["job_id"]
    run_ids = import_job(
        ctx.service.cp,
        job_id,
        store,
        experiment=ctx.opts.get("experiment") or "cloud",
        results_dir=(ctx.root / "results" / job_id).resolve(),
    )
    _print(ctx.opts, {"job_id": job_id, "runs": run_ids}, f"imported {len(run_ids)} run(s)")
    return run_ids


def _predict(ctx: _Context) -> Any:
    from logic.controllers.cloud.prediction import prediction_accuracy

    opts, service = ctx.opts, ctx.service
    action = opts["predict_command"]
    if action == "accuracy":
        report = prediction_accuracy(service.cp, opts["job_id"])
        text = f"{report['job_id']}: {len(report['tasks'])} task(s), median APE {report['median_ape']}, coverage {report['coverage']}"
        _print(opts, report, text)
        return report
    if action == "evaluate":
        from logic.controllers.cloud.prediction import default_predictor

        report = default_predictor().evaluate()
        _print(opts, report, " ".join(f"{k}={v}" for k, v in report.items()))
        return report
    if opts.get("manifest"):
        spec = load_manifest(Path(opts["manifest"]), policy=opts.get("policy"))
    else:
        spec = job_from_command(opts["sim_command"])
    spec.resources.machine_type = opts.get("machine_type") or spec.resources.machine_type
    predictions = [p.to_dict() for p in service.predict(spec) or []]
    total = sum(p["p50"] for p in predictions if p["p50"] is not None)
    text = "\n".join(
        f"{p['cell_id']}: {p['p50']} s ({p['p10']}-{p['p90']}), {p['memory_gb']} GiB [{p['source']}]"
        for p in predictions
    )
    payload = {"cells": predictions, "total_p50_seconds": round(total, 1)}
    _print(opts, payload, text + f"\ntotal (p50): {total:.0f} s")
    return payload


def _cost(ctx: _Context) -> Any:
    from logic.controllers.cloud.cost import import_billing_csv

    opts, service = ctx.opts, ctx.service
    action = opts["cost_command"]
    if action == "prices":
        prices = cast(Any, service.prices)  # None still fails at price_per_hour
        price = prices.price_per_hour(opts["provider"], opts["machine_type"], opts.get("region"), False)
        spot = prices.price_per_hour(opts["provider"], opts["machine_type"], opts.get("region"), True)
        payload = {
            "provider": opts["provider"],
            "machine_type": opts["machine_type"],
            "on_demand_usd_h": price,
            "spot_usd_h": spot,
        }
        _print(opts, payload, f"{opts['machine_type']}: on demand {price} USD/h, spot {spot} USD/h")
        return payload
    if action == "import-billing":
        totals = import_billing_csv(service.cp, Path(opts["path"]), opts["provider"], service.clock())
        _print(opts, {"jobs": totals}, f"recorded billing for {len(totals)} job(s)")
        return totals
    if action == "show":
        job = service.cp.get_job(opts["job_id"])
        tasks = service.cp.list_tasks(job.job_id)
        costs = service._task_costs(job, tasks)
        events = service.cp.list_events(job.job_id)
        billing = [e.data for e in events if e.kind == "billing"]
        payload = {
            "job_id": job.job_id,
            "accrued_usd": service.job_accrued(job, tasks),
            "budget_usd": job.spec.resources.budget_usd,
            "month_accrued_usd": service.month_accrued(service.clock()),
            "monthly_budget_usd": service.budget.monthly_usd if service.budget else None,
            "estimate": next((e.data for e in reversed(events) if e.kind == "cost_estimate"), None),
            "billing": billing[-1] if billing else None,
            "tasks": [
                {"task_id": t, "elapsed_s": c[0], "accrued_usd": c[1], "rate_usd_h": c[2]} for t, c in costs.items()
            ],
        }
        _print(opts, payload, f"{job.job_id}: {payload['accrued_usd']} USD accrued (cap {payload['budget_usd']})")
        return payload
    overrides = {k: v for k, v in _spec_overrides(opts).items() if v is not None}
    if opts.get("manifest"):
        spec = load_manifest(Path(opts["manifest"]), policy=opts.get("policy"), **overrides)
    else:
        spec = job_from_command(opts["sim_command"], **overrides)
    predictions = service.predict(spec)
    estimate = service.estimate(spec, predictions) if predictions is not None else None
    if estimate is None:
        _print(opts, None, "no estimate (no runtime predictor)")
        return None
    predictions = cast(Any, predictions)  # None still fails when the cells are read
    payload = estimate.to_dict()
    payload.update(
        cost_usd=estimate.mid,
        eta_seconds=_makespan([p.p50 for p in predictions if p.p50 is not None], spec.max_parallel),
        memory_required_gb=max((p.memory_gb for p in predictions), default=None),
        hourly_rate_usd=estimate.price_per_hour,
        provider=Provider(spec.provider).value,
        compute=ComputeKind(spec.compute).value,
        source=f"CR-10 estimate: p50 of {len(estimate.cells)} cell(s), range {estimate.low}-{estimate.high} USD",
    )
    _print(opts, payload, f"{estimate.mid} USD ({estimate.low}-{estimate.high}) at {estimate.price_per_hour} USD/h")
    return payload


def _makespan(seconds: List[float], slots: int) -> Optional[float]:
    """Longest-first list scheduling of the cells' p50 times onto ``slots`` parallel tasks."""
    if not seconds:
        return None
    loads = [0.0] * max(1, slots)
    for value in sorted(seconds, reverse=True):
        loads[loads.index(min(loads))] += value
    return round(max(loads), 1)


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
    if opts.get("pool"):
        from logic.controllers.cloud.scheduler import run_pool_worker

        worker = pool_worker_service(ctx.root, ctx.project_root, slots=int(opts.get("slots") or 1))
        reason = run_pool_worker(worker, idle_seconds=float(opts.get("idle") or 300.0), poll_seconds=poll)
        _print(opts, {"stopped": reason}, reason)
        return reason
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
    "metrics": _metrics,
    "track": _track,
    "predict": _predict,
    "cost": _cost,
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
    if opts.get("profile"):
        from logic.controllers.cloud.auth.config import load_profiles
        from logic.controllers.cloud.profile_service import profile_service

        profiles = load_profiles()
        if opts["profile"] not in profiles:
            raise ValueError(
                f"unknown cloud profile {opts['profile']!r} (known: {', '.join(sorted(profiles)) or 'none'})"
            )
        opts["_profile"] = profiles[opts["profile"]]
        service = profile_service(opts["_profile"], root, project_root)
    else:
        service = local_service(root, project_root)
    return _COMMANDS[command](_Context(opts, root, project_root, service))
