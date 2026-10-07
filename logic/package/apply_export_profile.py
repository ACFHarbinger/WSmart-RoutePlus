"""Apply an export profile to a package tree.

A profile (``export_profiles`` in ``ci/export_config.json``) describes the changes that turn an
export tree into a narrower package, so a re-cut reproduces them without hand edits:

1. ``remove_paths``: files and directories to delete.
2. ``entrypoints``: which of ``main.py`` / ``__main__.py`` to keep (``both``, ``main``, ``dunder``,
   ``none``); overridable with ``--entrypoints``.
3. ``patch``: a unified diff (relative to the tree after step 1) with the edits to the files that
   stay, applied with ``git apply --3way`` when the tree is a git checkout.
4. ``engines``: per-policy solver frameworks to keep; overridable with ``--engines``. Each engine
   in ``policy_engines`` lists its files and dependencies; unselected engines lose both, and a
   policy's ``default_framework`` settings are pointed at the first kept engine.
5. A static check that no kept module imports a removed one.
6. Optionally (``--file-list PATH``), a ``source/`` folder with the entry points and ``logic/``
   and its ``tree`` listing saved to ``PATH``.

Usage (from the main repository, against a re-cut export checkout)::

    python logic/package/apply_export_profile.py --root ../wsr-package/wt --profile minimal-ptp
    python logic/package/apply_export_profile.py --root ../wsr-package/wt --profile minimal-ptp \\
        --entrypoints both --engines swc_tcf=gurobi,ortools

Without ``--profile`` only the entry-point and engine options are applied. The default engine
selection is the profile's ``engines`` entry; without one, every engine is kept.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "ci" / "export_config.json"
_ENTRYPOINTS = {"both": ("main.py", "__main__.py"), "main": ("main.py",), "dunder": ("__main__.py",), "none": ()}


def _log(msg: str) -> None:
    print(f"[export-profile] {msg}")


def _remove(root: Path, rel: str, dry_run: bool) -> None:
    path = root / rel
    if path.is_symlink():
        raise SystemExit(f"refusing to follow a symlink: {rel}")
    if not path.exists():
        return
    _log(f"remove {rel}")
    if dry_run:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def apply_entrypoints(root: Path, choice: str, dry_run: bool) -> None:
    """Keep only the requested entry-point scripts at the package root."""
    if choice not in _ENTRYPOINTS:
        raise SystemExit(f"--entrypoints must be one of {sorted(_ENTRYPOINTS)}")
    for name in ("main.py", "__main__.py"):
        if name not in _ENTRYPOINTS[choice]:
            _remove(root, name, dry_run)


def apply_patch(root: Path, patch: Path, dry_run: bool) -> None:
    """Apply the profile's edits to the kept files (3-way when the tree is a git checkout)."""
    if not patch.exists():
        raise SystemExit(f"profile patch not found: {patch}")
    is_git = subprocess.run(["git", "-C", str(root), "rev-parse"], capture_output=True).returncode == 0
    cmd = ["git", "apply", "--whitespace=nowarn"] + (["--3way"] if is_git else []) + (["--check"] if dry_run else [])
    _log(f"apply {patch.name}")
    result = subprocess.run(cmd + [str(patch.resolve())], cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"patch failed:\n{result.stderr}")


def _parse_engines(spec: Optional[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for part in filter(None, (spec or "").split(";")):
        policy, _, engines = part.partition("=")
        out[policy.strip()] = [e.strip() for e in engines.split(",") if e.strip()]
    return out


def apply_engines(root: Path, catalogue: Dict, selection: Dict[str, List[str]], dry_run: bool) -> None:
    """Drop the files and dependencies of every engine that is not selected."""
    for policy, engines in catalogue.items():
        if policy.startswith("_"):
            continue
        keep = selection.get(policy, list(engines["engines"]))
        unknown = set(keep) - set(engines["engines"])
        if unknown or not keep:
            raise SystemExit(f"{policy}: unknown or empty engine selection {sorted(unknown) or keep}")
        kept_deps = {d for e in keep for d in engines["engines"][e].get("dependencies", [])}
        for name, spec in engines["engines"].items():
            if name in keep:
                continue
            _log(f"{policy}: drop engine {name}")
            for rel in spec.get("files", []):
                _remove(root, rel, dry_run)
            for dep in set(spec.get("dependencies", [])) - kept_deps:
                for pyproject in engines.get("pyprojects", ["logic/pyproject.toml"]):
                    _edit(root / pyproject, rf'\n[ \t]*"{re.escape(dep)}[<>=!~][^"\n]*",', "", dry_run)
        default = engines.get("default_framework")
        if default and default["value"] not in keep:
            for rewrite in default["rewrites"]:
                for rel in rewrite["files"]:
                    _edit(root / rel, rewrite["pattern"].format(value=re.escape(default["value"])),
                          rewrite["replacement"].format(value=keep[0]), dry_run)


def _edit(path: Path, pattern: str, repl: str, dry_run: bool) -> None:
    if not path.exists():
        return
    text = path.read_text()
    new = re.sub(pattern, repl, text)
    if new != text:
        _log(f"edit {path.name}")
        if not dry_run:
            path.write_text(new)


def write_file_list(root: Path, target: str, dry_run: bool) -> Path:
    """Copy the entry points and logic/ into ``root/source`` and save ``tree`` run there to ``target``.

    ``target`` is relative to ``root`` (e.g. ``assets/files/FILE_LIST.txt``). ``source/`` is
    recreated on every call; ``__pycache__`` folders are not copied.
    """
    source = root / "source"
    out = root / target
    _log(f"file list: source/ -> {target}")
    if dry_run:
        return out
    if source.is_symlink():
        raise SystemExit("refusing to replace a symlinked source/")
    if source.exists():
        shutil.rmtree(source)
    source.mkdir()
    for name in ("main.py", "__main__.py"):
        if (root / name).is_file():
            shutil.copy2(root / name, source / name)
    shutil.copytree(root / "logic", source / "logic", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    try:
        listing = subprocess.run(
            ["tree", "--charset", "ascii", "--noreport", "-a", "."], cwd=source, capture_output=True, text=True, check=True
        ).stdout
    except FileNotFoundError as exc:
        raise SystemExit("the 'tree' command is required for --file-list") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(listing)
    return out


def check_imports(root: Path) -> List[str]:
    """Return 'file: module' for every import of a logic.* module that no longer exists."""
    problems = []
    for py in (root / "logic").rglob("*.py"):
        try:
            tree = ast.parse(py.read_text(errors="ignore"))
        except SyntaxError as exc:
            problems.append(f"{py.relative_to(root)}: syntax error {exc}")
            continue
        guarded = {
            id(n)
            for t in ast.walk(tree)
            if isinstance(t, ast.Try) and any("ImportError" in ast.unparse(h.type or ast.Name("")) for h in t.handlers)
            for n in ast.walk(t)
        }
        for node in ast.walk(tree):
            if id(node) in guarded:
                continue  # optional import with an ImportError fallback
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for mod in mods:
                if not mod.startswith("logic."):
                    continue
                base = root.joinpath(*mod.split("."))
                if not (base.with_suffix(".py").exists() or (base / "__init__.py").exists()):
                    problems.append(f"{py.relative_to(root)}: {mod}")
    return problems


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=_PROJECT_ROOT, help="Tree to modify (default: this repository).")
    p.add_argument("--config", type=Path, default=_CONFIG_PATH)
    p.add_argument("--profile", default=None, help="Name under export_profiles in the config (optional).")
    p.add_argument("--entrypoints", choices=sorted(_ENTRYPOINTS), default=None,
                   help="Override the profile: keep main.py, __main__.py, both or neither.")
    p.add_argument("--engines", default=None,
                   help="Override the profile, e.g. 'swc_tcf=gurobi;sans=new'. Policies left out keep the profile choice.")
    p.add_argument("--file-list", metavar="PATH", default=None,
                   help="Create source/ (entry points + logic/) and save its tree listing to PATH (relative to --root).")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    config = json.loads(args.config.read_text())
    profile = {}
    if args.profile is not None:
        profile = config.get("export_profiles", {}).get(args.profile)
        if profile is None:
            raise SystemExit(f"unknown profile {args.profile!r}; known: {sorted(config.get('export_profiles', {}))}")
    root = args.root.resolve()

    for rel in profile.get("remove_paths", []):
        _remove(root, rel, args.dry_run)
    apply_entrypoints(root, args.entrypoints or profile.get("entrypoints", "both"), args.dry_run)
    if profile.get("patch"):
        apply_patch(root, args.config.parent / profile["patch"], args.dry_run)
    selection = _parse_engines(profile.get("engines"))
    selection.update(_parse_engines(args.engines))
    apply_engines(root, config.get("policy_engines", {}), selection, args.dry_run)

    if not args.dry_run:
        problems = check_imports(root)
        if problems:
            print("\n".join(problems))
            raise SystemExit(f"{len(problems)} import(s) of removed modules")
    if args.file_list:
        write_file_list(root, args.file_list, args.dry_run)
    _log("done")


if __name__ == "__main__":
    sys.exit(main())
