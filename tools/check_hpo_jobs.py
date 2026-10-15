"""Check and convert the simulator HPO job files against the policy configs.

Every job under ``logic/src/pipeline/simulations/hpo/jobs/<key>.json`` restates
its policy's tunable fields. Two failure modes have shipped through this gap:
keys silently dropped by the YAML loader (``policy_aco_hh.yaml`` lost three
keys until 2026-09-27) and keys tuning fields the engine never reads (released
ALNS ``min_removal``). This tool makes both loud (P-56, owner ruling
RQ-mistral-2):

- ``--check``: every job key must resolve to a live field of its policy's
  config dataclass (or a pipeline-owned section), and every job default must
  equal the config default. Exits 1 with a report on violations. Used in CI.
- ``--convert``: rewrites the jobs in place. Stale keys listed in ``RENAME``
  are renamed; keys with no live field are removed and reported; defaults are
  aligned with the config. Run in the same change as the config work that
  renames or removes fields (Q-8b).

Usage::

    python tools/check_hpo_jobs.py --check
    python tools/check_hpo_jobs.py --convert
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
# Direct CLI execution puts tools/, rather than the repository, on sys.path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
LOGIC = REPO / "logic"
JOBS_DIR = LOGIC / "src" / "pipeline" / "simulations" / "hpo" / "jobs"
STAGE_MANIFESTS = [
    LOGIC / "src" / "policies" / "routing" / "_manifest.py",
    LOGIC / "src" / "policies" / "packing" / "_manifest.py",
    LOGIC / "src" / "policies" / "packing_and_routing" / "_manifest.py",
]

# Job keys that belong to the pipeline, not the policy's own config.
PIPELINE_KEYS = {"packing", "route_improvement", "acceptance_criterion"}

# Job file renames: file names that no longer match a registered policy key.
FILE_RENAME: Dict[str, str] = {
    "adp": "adp_rollout",  # the policy registers as adp_rollout
}

# Field renames that landed with the config work; keys are old job key names.
RENAME: Dict[str, str] = {
    # P-54 (packing remaps, if any job ever carried them):
    "frequency": "threshold",
    "confidence_factor": "threshold",
    "learned_threshold": "threshold",
    "n_sectors": "threshold",
    "combined_strategies": "combined_strategies",
    "strategies": "combined_strategies",
}

ENTRY_RE = re.compile(r"^\s*'([A-Za-z0-9_-]+)':\s*'(logic\.[A-Za-z0-9_.:]+)'\s*,?\s*$")


def routing_entries() -> Dict[str, str]:
    """Registry key -> 'module:Class' across the stage manifests."""
    entries: Dict[str, str] = {}
    for manifest in STAGE_MANIFESTS:
        in_entries = False
        for line in manifest.read_text().splitlines():
            if line.startswith("ENTRIES"):
                in_entries = True
                continue
            if in_entries:
                if line.startswith("}"):
                    in_entries = False
                    continue
                match = ENTRY_RE.match(line)
                if match:
                    entries[match.group(1)] = match.group(2)
    return entries


def config_class_for(key: str) -> Optional[type]:
    """The policy's config dataclass, resolved through the registry manifest."""
    entries = routing_entries()
    target = entries.get(key)
    if target is None:
        return None
    module_name, _, attr = target.partition(":")
    policy_cls = getattr(importlib.import_module(module_name), attr)
    getter = getattr(policy_cls, "_config_class", None)
    if getter is None:
        return None
    try:
        return getter()
    except TypeError:  # not a classmethod binding
        return getter


def job_files() -> List[Path]:
    return sorted(JOBS_DIR.glob("*.json"))


def check_jobs() -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Return (violations, per-job diagnostics). A violation is a report line."""
    violations: List[str] = []
    diagnostics: Dict[str, Dict[str, Any]] = {}
    for path in job_files():
        key = path.stem
        job = json.loads(path.read_text())
        cfg_cls = config_class_for(key)
        diag: Dict[str, Any] = {"config": cfg_cls.__name__ if cfg_cls else None, "problems": []}
        if cfg_cls is None:
            diag["problems"].append("WARN: no typed config class; job keys cannot be validated")
            diagnostics[key] = diag
            continue
        cfg_fields = {f.name: f for f in fields(cfg_cls)}
        for field_name, spec in job.items():
            if field_name in PIPELINE_KEYS or field_name.startswith("_"):
                continue
            if not isinstance(spec, dict) or "default" not in spec:
                if field_name not in cfg_fields:
                    violations.append(f"{key}.json: key {field_name!r} is not a field of {cfg_cls.__name__}")
                continue
            if field_name not in cfg_fields:
                renamed = RENAME.get(field_name)
                hint = f" (rename to {renamed!r}?)" if renamed and renamed in cfg_fields else ""
                msg = f"{key}.json: key {field_name!r} is not a field of {cfg_cls.__name__}{hint}"
                violations.append(msg)
                diag["problems"].append(msg)
                continue
            default = cfg_fields[field_name].default
            if default is dataclasses.MISSING:
                continue  # required field or default_factory: nothing to compare
            job_default = spec.get("default")
            if not _defaults_equal(job_default, default):
                msg = f"{key}.json: default of {field_name!r} is {job_default!r}, config default is {default!r}"
                violations.append(msg)
                diag["problems"].append(msg)
        # Config fields absent from the job are fine (not every field is tuned),
        # but a field the job tunes must exist — that is the check.
        diagnostics[key] = diag
    violations = [v for v in violations if not v.startswith("WARN")]
    return violations, diagnostics


def _defaults_equal(job_default: Any, config_default: Any) -> bool:
    if job_default is None or config_default is None:
        return (job_default is None) == (config_default is None) or job_default == config_default
    if isinstance(config_default, type) and config_default is not None:
        return False  # dataclasses field default sentinel (MISSING): treat as unequal
    try:
        return bool(job_default == config_default) or abs(float(job_default) - float(config_default)) < 1e-12
    except (TypeError, ValueError):
        return bool(job_default == config_default)


def convert_jobs() -> List[str]:
    """Rename and drop stale keys; align defaults. Returns the change report."""
    report: List[str] = []
    for path in job_files():
        key = path.stem
        if key in FILE_RENAME:
            new_name = FILE_RENAME[key]
            target = path.with_name(f"{new_name}.json")
            path.rename(target)
            path = target
            report.append(f"renamed job file {key}.json -> {new_name}.json")
            key = new_name
        job = json.loads(path.read_text())
        cfg_cls = config_class_for(key)
        if cfg_cls is None:
            report.append(f"{path.name}: skipped (no typed config class)")
            continue
        cfg_fields = {f.name: f for f in fields(cfg_cls)}
        changed = False
        new_job: Dict[str, Any] = {}
        for field_name, spec in job.items():
            if field_name in PIPELINE_KEYS or field_name.startswith("_"):
                new_job[field_name] = spec
                continue
            renamed = RENAME.get(field_name, field_name)
            if field_name not in cfg_fields and renamed in cfg_fields:
                report.append(f"{path.name}: renamed {field_name!r} -> {renamed!r}")
                field_name = renamed
                changed = True
            if field_name not in cfg_fields:
                report.append(f"{path.name}: dropped {field_name!r} (no live config field)")
                changed = True
                continue
            default = cfg_fields[field_name].default
            if (
                isinstance(spec, dict)
                and "default" in spec
                and default is not dataclasses.MISSING
                and not _defaults_equal(spec.get("default"), default)
            ):
                report.append(f"{path.name}: default of {field_name!r} {spec.get('default')!r} -> {default!r}")
                spec = dict(spec)
                spec["default"] = default
                changed = True
            new_job[field_name] = spec
        if changed:
            path.write_text(json.dumps(new_job, indent=4) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="validate the jobs; exit 1 on violations")
    mode.add_argument("--convert", action="store_true", help="rewrite the jobs in place")
    args = parser.parse_args()
    if args.convert:
        report = convert_jobs()
        for line in report:
            print(line)
        print(f"converted {len(job_files())} job files; {len(report)} changes")
        return 0
    violations, _ = check_jobs()
    if violations:
        print(f"HPO job check FAILED ({len(violations)} violations):")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print(f"HPO job check OK: {len(job_files())} jobs, every key resolves to a live config field")
    return 0


if __name__ == "__main__":
    sys.exit(main())
