"""Pruning of the staged release tree: registry manifests, lazy maps and package initialisers trimmed to the closure, plus the post-build checks.

Attributes:
    filter_manifest_file: Keep only the released keys of a registry manifest.
    filter_lazy_map: Drop lazy-map entries whose modules are not staged.
    drop_pruned_init_imports: Remove ``__init__`` imports of pruned modules.
    verify_staged: Every staged import resolves inside the staged tree.

Example:
    >>> problems = verify_staged(out, staged_modules)
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Set, Tuple

from logic.package.release_closure import (
    ENTRY_RE,
    REPO,
    _package_of,
    imports_of,
    module_to_file,
    parse_registry_manifest,
    trimmed,
)

DECLARE_RE = re.compile(r"""\.declare\(['"]([A-Za-z0-9_-]+)['"],\s*ENTRIES\[['"]([A-Za-z0-9_-]+)['"]]\)""")


def filter_manifest_file(path: Path, released: Set[str]) -> List[str]:
    """Keep only released entries in a staged registry manifest."""
    dropped: List[str] = []
    lines = path.read_text().splitlines(keepends=True)
    out: List[str] = []
    in_entries = False
    for line in lines:
        if line.startswith("ENTRIES"):
            in_entries = True
            out.append(line)
            continue
        if in_entries and line.startswith("}"):
            in_entries = False
            out.append(line)
            continue
        if in_entries:
            match = ENTRY_RE.match(line.rstrip("\n"))
            if match and match.group(1) not in released:
                dropped.append(f"{path.name}: key {match.group(1)}")
                continue
            out.append(line)
            continue
        declare = DECLARE_RE.search(line)
        if declare and declare.group(2) not in released:
            dropped.append(f"{path.name}: declare {declare.group(1)}")
            continue
        out.append(line)
    path.write_text("".join(out))
    return dropped


def filter_lazy_map(path: Path, staged_modules: Set[str], out: Path) -> List[str]:
    """Drop pruned lazy targets regardless of quoting or wrapping in the source map."""
    source = path.read_text()
    for node in ast.parse(source).body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if not any(isinstance(target, ast.Name) and target.id == "_LAZY" for target in targets):
            continue
        try:
            mapping = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if not isinstance(mapping, dict):
            continue
        kept = {}
        dropped = []
        for key, value in mapping.items():
            if not isinstance(value, str):
                kept[key] = value
                continue
            target = value.split(":")[0]
            present = (
                target in staged_modules
                if module_to_file(target, REPO) is not None
                else trimmed(target, out) in staged_modules
            )
            if present or not target.startswith("logic."):
                kept[key] = value
            else:
                dropped.append(f"{path.relative_to(out)}: lazy {key} -> {target}")
        lines = source.splitlines(keepends=True)
        lines[node.lineno - 1 : node.end_lineno] = ["_LAZY = " + repr(kept) + "\n"]
        path.write_text("".join(lines))
        return dropped
    return []


def drop_pruned_init_imports(path: Path, staged_modules: Set[str], out: Path) -> List[str]:
    """In a staged __init__.py, remove module-level imports of pruned modules."""
    dropped: List[str] = []
    root = out
    tree = ast.parse(path.read_text())
    remove_spans: List[Tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = (node.module or "").split(".")
                targets = [".".join(base + [alias.name]) for alias in node.names]
                if node.module:
                    targets.append(node.module)
            else:
                pkg = _package_of(path, root)
                up = pkg[: len(pkg) - (node.level - 1)]
                base = up + ([node.module] if node.module else [])
                targets = [".".join(base + [alias.name]) for alias in node.names]
                targets.append(".".join(base))
        else:
            continue
        for target in targets:
            if not target.startswith("logic"):
                continue
            # A target that maps to a dev file (submodule import) must exist
            # staged under its full name; anything else resolves by prefix.
            if module_to_file(target, REPO) is not None:
                resolved = target if target in staged_modules else ""
            else:
                resolved = trimmed(target, out)
            if resolved not in staged_modules:
                remove_spans.append((node.lineno, node.end_lineno))
                dropped.append(f"{path.relative_to(out)}: import {target}")
                break
    if remove_spans:
        lines = path.read_text().splitlines(keepends=True)
        for start, end in sorted(remove_spans, reverse=True):
            del lines[start - 1 : end]
        path.write_text("".join(lines))
    return dropped


def validate_release_tests(out: Path, staged_tests: List[Path]) -> Tuple[List[Path], List[str]]:
    """Keep only release tests that collect cleanly against the staged tree.

    The static selection (imports inside the closure) cannot see dynamically
    constructed imports, so the staged copy is authoritative: pytest collects
    the staged test files and the ones that fail are excluded from the release
    set (with their paths recorded). Falls back to the static selection when
    pytest is unavailable.
    """
    if not staged_tests:
        return [], []
    import os

    rel_paths = [str(path.relative_to(out)) for path in staged_tests]
    env = dict(os.environ, PYTHONPATH=str(out))
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--collect-only",
                "--continue-on-collection-errors",
                *rel_paths,
            ],
            cwd=out,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return staged_tests, []
    failed = sorted(
        line.removeprefix("ERROR ").strip()
        for line in result.stdout.splitlines()
        if line.startswith("ERROR ") and line.strip().endswith(".py")
    )
    failed_set = set(failed)
    kept = [path for path in staged_tests if str(path.relative_to(out)) not in failed_set]
    return kept, failed


def verify_staged(out: Path, staged_modules: Set[str]) -> List[str]:
    """Every module-level import and manifest target must stay inside the tree."""
    errors: List[str] = []
    for path in (out / "logic").rglob("*.py"):
        imports = imports_of(path, out)
        for dep in sorted(imports.module_level):
            root = trimmed(dep, out)
            if root and root not in staged_modules:
                errors.append(f"{path.relative_to(out)}: module-level import {dep} escapes the tree")
            if module_to_file(dep, REPO) is not None and dep not in staged_modules:
                errors.append(f"{path.relative_to(out)}: module-level import {dep} targets a pruned submodule")
        for dep in sorted(imports.deferred):
            root = trimmed(dep, out)
            if root and root not in staged_modules:
                errors.append(
                    f"{path.relative_to(out)}: deferred import {dep} of a pruned module "
                    "(code path unavailable in the release)"
                )
    for staged_manifest in (out / "logic").rglob("_manifest.py"):
        for key, target in parse_registry_manifest(staged_manifest).items():
            if trimmed(target.split(":")[0], out) not in staged_modules:
                errors.append(f"{staged_manifest.name}: key {key!r} target {target} escapes the tree")
    return errors
