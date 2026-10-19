"""Inputs of remote tasks: staged to object storage at submit, fetched by the task runtime.

At submit, :func:`stage_job_inputs` uploads the job's shared inputs (``JobSpec.inputs``,
by default ``data/simulator``) and each cell's policy YAML directory, content-addressed
so unchanged files are uploaded once across jobs, and writes a manifest:

.. code-block:: json

    {"shared": {"data/simulator/datasets/x.npz": "cas/<sha256>", ...},
     "cells": {"<cell_id>": {"policy_aco_hh.yaml": "cas/<sha256>"}}}

The runtime downloads the shared files under the project root (where ``main.py``
resolves ``data/...``) and the cell's policy files into its work directory, checking
every hash.

Attributes:
    stage_job_inputs: Upload a job's inputs and return the manifest key.
    fetch_inputs: Download one cell's inputs into a work directory.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Optional

from logic.controllers.cloud.interfaces import ObjectStore
from logic.controllers.cloud.model import JobSpec
from logic.controllers.cloud.storage import sha256_file


def _upload_tree(store: ObjectStore, base: Path, relroot: Path) -> Dict[str, str]:
    """Upload every file under ``base / relroot`` content-addressed; keys are paths relative to ``base``."""
    target = base / relroot
    files = [target] if target.is_file() else sorted(p for p in target.rglob("*") if p.is_file())
    entries = {}
    for path in files:
        entries[path.relative_to(base).as_posix()] = store.put_content_addressed(path).key
    return entries


def stage_job_inputs(store: ObjectStore, spec: JobSpec, project_root: Path, job_id: str) -> str:
    """Upload a job's shared inputs and policy directories; returns the manifest's key.

    Raises:
        FileNotFoundError: A listed input or a cell's policy directory does not exist.
    """
    project_root = Path(project_root)
    shared: Dict[str, str] = {}
    for relpath in spec.inputs:
        if not (project_root / relpath).exists():
            raise FileNotFoundError(f"job input {relpath!r} not found under {project_root}")
        shared.update(_upload_tree(store, project_root, Path(relpath)))
    cells: Dict[str, Dict[str, str]] = {}
    for cell in spec.cells:
        if cell.policy_dir:
            policy = Path(cell.policy_dir)
            if not policy.is_dir():
                raise FileNotFoundError(f"cell {cell.cell_id!r}: policy directory {policy} not found")
            cells[cell.cell_id] = _upload_tree(store, policy, Path("."))
    key = f"inputs/{job_id}/manifest.json"
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "manifest.json"
        manifest.write_text(json.dumps({"shared": shared, "cells": cells}, sort_keys=True))
        store.put_file(manifest, key)
    return key


def fetch_inputs(
    store: ObjectStore, manifest_key: str, cell_id: str, project_root: Path, workdir: Path
) -> Optional[Path]:
    """Download the shared inputs under ``project_root`` and the cell's policy files into ``workdir/policy``.

    Returns:
        The policy directory, or None if the cell has no policy files.

    Raises:
        ValueError: A downloaded file does not match its content address.
    """
    workdir = Path(workdir)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "manifest.json"
        store.get_file(manifest_key, local)
        manifest = json.loads(local.read_text())
    _fetch(store, manifest.get("shared", {}), Path(project_root))
    policy_files = manifest.get("cells", {}).get(cell_id)
    if not policy_files:
        return None
    policy = workdir / "policy"
    _fetch(store, policy_files, policy)
    return policy


def _fetch(store: ObjectStore, entries: Dict[str, str], dest: Path) -> None:
    for relpath, key in entries.items():
        target = dest / relpath
        if target.is_file() and key.endswith(sha256_file(target)):
            continue  # already present (a warm worker, or a retry on the same machine)
        store.get_file(key, target)
        if not key.endswith(sha256_file(target)):
            raise ValueError(f"input {relpath} does not match {key}")
