"""Build job specs from manifests and single commands.

Two manifest formats are read:

- **native**: ``{"name": ..., "cells": [{"cell_id", "argv" | "command", "policy_dir",
  "expected_wall_seconds", "needs_gurobi", "env"}], ...JobSpec fields}``;
- **rerun**: the paper rerun manifests, whose cells carry a shell ``command``
  (``WSR_POLICY_CONFIG_DIR=<dir> python main.py test_sim ...``), ``cell_id``,
  ``expected_wall_seconds`` and ``needs_gurobi``. A ``policy`` filter keeps one
  routing algorithm's cells.

Attributes:
    parse_command: Split a ``test_sim`` command line into argv, policy dir and env.
    cell_from_command: One cell from one command line.
    load_manifest: A job spec from a manifest file.
    job_from_command: A one-cell job spec from one command line.

Example:
    >>> cell = cell_from_command("test_sim sim.graph.num_loc=100", cell_id="c1")
    >>> cell.argv
    ['test_sim', 'sim.graph.num_loc=100']
"""

from __future__ import annotations

import json
import os
import re
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from logic.controllers.cloud.model import (
    CellSpec,
    ComputeKind,
    GurobiMode,
    JobSpec,
    PlacementTarget,
    Provider,
    Resources,
)

_ENV_ASSIGNMENT = re.compile(r"^[A-Z_][A-Z0-9_]*=")
# Cluster-wide settings the runner owns; a manifest's own values are dropped.
_RUNNER_OWNED = ("sim.cpu_cores=", "sim.resume=", "sim.checkpoint_days=", "sim.output_dir=")


def parse_command(command: str) -> Tuple[List[str], Optional[str], Dict[str, str]]:
    """Split a ``test_sim`` command line.

    Accepts ``test_sim ...``, ``main.py test_sim ...`` and ``python main.py test_sim ...``,
    with leading ``NAME=value`` environment assignments.

    Args:
        command: The command line.

    Returns:
        ``(argv, policy_dir, env)``: arguments from ``test_sim`` on, the
        ``WSR_POLICY_CONFIG_DIR`` value if given, and any other environment assignments.

    Raises:
        ValueError: The command does not run ``test_sim``.
    """
    tokens = shlex.split(command)
    env: Dict[str, str] = {}
    while tokens and _ENV_ASSIGNMENT.match(tokens[0]):
        name, _, value = tokens.pop(0).partition("=")
        env[name] = value
    if "test_sim" not in tokens:
        raise ValueError(f"not a test_sim command: {command!r}")
    argv = tokens[tokens.index("test_sim") :]
    argv = [arg for arg in argv if not arg.startswith(_RUNNER_OWNED)]
    policy_dir = env.pop("WSR_POLICY_CONFIG_DIR", None)
    return argv, policy_dir, env


def cell_from_command(
    command: str,
    cell_id: Optional[str] = None,
    expected_wall_seconds: Optional[float] = None,
    needs_gurobi: bool = False,
) -> CellSpec:
    """One cell from one command line (see :func:`parse_command`)."""
    argv, policy_dir, env = parse_command(command)
    return CellSpec(
        cell_id=cell_id or _default_cell_id(argv),
        argv=argv,
        policy_dir=os.path.expandvars(policy_dir) if policy_dir else None,
        env=env,
        expected_wall_seconds=expected_wall_seconds,
        needs_gurobi=needs_gurobi,
    )


def _default_cell_id(argv: List[str]) -> str:
    """``<run_name>/<policies>`` from the overrides, or ``cell``."""
    values = dict(arg.split("=", 1) for arg in argv[1:] if "=" in arg)
    parts = [values.get("sim.run_name"), values.get("sim.policies")]
    parts = [part.strip("'\"[]") for part in parts if part]
    return "/".join(parts) or "cell"


def _cells(data: Dict[str, Any], policy: Optional[str]) -> List[CellSpec]:
    cells = []
    for index, raw in enumerate(data.get("cells", [])):
        if policy is not None and raw.get("policy") != policy:
            continue
        cell_id = raw.get("cell_id") or f"cell-{index}"
        if "argv" in raw:
            cells.append(
                CellSpec(
                    cell_id=cell_id,
                    argv=list(raw["argv"]),
                    policy_dir=raw.get("policy_dir"),
                    env=dict(raw.get("env", {})),
                    expected_wall_seconds=raw.get("expected_wall_seconds"),
                    needs_gurobi=bool(raw.get("needs_gurobi", False)),
                )
            )
            continue
        cell = cell_from_command(
            raw["command"],
            cell_id=cell_id,
            expected_wall_seconds=raw.get("expected_wall_seconds"),
            needs_gurobi=bool(raw.get("needs_gurobi", False)),
        )
        if raw.get("policy_dir"):
            cell.policy_dir = raw["policy_dir"]
        cells.append(cell)
    return cells


def _job_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    if "provider" in data:
        fields["provider"] = Provider(data["provider"])
    if "compute" in data:
        fields["compute"] = ComputeKind(data["compute"])
    if "resources" in data:
        fields["resources"] = Resources(**data["resources"])
    if "gurobi" in data:
        fields["gurobi"] = GurobiMode(data["gurobi"])
    for name in ("max_parallel", "gurobi_secret", "image", "labels", "priority", "pool"):
        if name in data:
            fields[name] = data[name]
    if "placement" in data:  # CR-09: [{"provider": "gcp", "compute": "batch", "max_tasks": 8, ...}, ...]
        fields["placement"] = [
            PlacementTarget(**{**t, "provider": Provider(t["provider"]), "compute": ComputeKind(t["compute"])})
            for t in data["placement"]
        ]
    return fields


def load_manifest(path: Path, policy: Optional[str] = None, **overrides: Any) -> JobSpec:
    """A job spec from a native or rerun manifest file (JSON or YAML).

    Args:
        path: The manifest file.
        policy: Keep only the cells whose ``policy`` equals this (rerun manifests).
        **overrides: :class:`JobSpec` fields that replace the manifest's.

    Returns:
        The validated job spec.
    """
    text = Path(path).read_text()
    if Path(path).suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    fields: Dict[str, Any] = {"name": data.get("name") or Path(path).stem, "cells": _cells(data, policy)}
    fields.update(_job_fields(data))
    fields.update({key: value for key, value in overrides.items() if value is not None})
    spec = JobSpec(**fields)
    spec.validate()
    return spec


def job_from_command(command: str, name: Optional[str] = None, **overrides: Any) -> JobSpec:
    """A one-cell job spec from one command line."""
    cell = cell_from_command(command)
    fields: Dict[str, Any] = {"name": name or cell.cell_id, "cells": [cell]}
    fields.update({key: value for key, value in overrides.items() if value is not None})
    spec = JobSpec(**fields)
    spec.validate()
    return spec
