"""WSmart-Route Algorithm Export Packager.

End-to-end pipeline that:

1. Validates user-requested algorithm acronyms against ``ci/export_config.json``.
2. Creates a timestamped git branch (``algo-export-YYYYMMDD_HHMMSS``).
3. Prunes all unselected algorithms from the working tree via
   ``logic/package/prune_codebase.py``.
4. Optionally drops pipeline features (META_LEARNING, HPO, EVAL, SECURITY, …).
5. Writes a tree listing of retained ``logic/src/`` files to
   ``retained_logic_files.txt``.
6. Builds standalone executables via ``pyinstaller ci/simulator.spec --clean``.
7. Packages the pruned source tree, the executables, and the tree listing
   into a timestamped ``.zip`` archive (excluding ``.git/``,
   ``__pycache__/``, and object files).
8. On any failure: aborts, deletes the export branch, and restores the
   original branch (rollback).

Usage::

    python ci/packager.py \\
        --constructors HGS,ALNS,BPC \\
        --selectors MS_REGULAR,MS_PPK \\
        --improvement RI_OROPT,RI_S2OPT \\
        --acceptance AC_GD,AC_OI \\
        --models AM,TAM \\
        --rl-algorithms RL_REINFORCE,RL_PPO \\
        --drop-features META_LEARNING,HPO,SECURITY \\
        [--dry-run] \\
        [--skip-build] \\
        [--output-dir dist/exports]

Acronyms may be comma-separated or space-separated (or both).
Omitting a category flag keeps every algorithm in that category.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Project root — packager.py lives at ci/, one level below the repo root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # WSmart-Route/
PRUNE_SCRIPT = PROJECT_ROOT / "logic" / "package" / "prune_codebase.py"
PROFILE_SCRIPT = PROJECT_ROOT / "logic" / "package" / "apply_export_profile.py"
SPEC_FILE = PROJECT_ROOT / "ci" / "simulator.spec"
CONFIG_FILE = PROJECT_ROOT / "ci" / "export_config.json"
LOGIC_SRC = PROJECT_ROOT / "logic" / "src"
LOGIC_DIR = PROJECT_ROOT / "logic"

_ZIP_EXCLUDE_DIRS: Set[str] = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".venv"}
_ZIP_EXCLUDE_SUFFIXES: Set[str] = {".o", ".pyc", ".pyo"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[packager] {msg}", flush=True)


def _die(msg: str, code: int = 1) -> None:
    print(f"[packager] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _load_config() -> Dict:
    if not CONFIG_FILE.exists():
        _die(f"export_config.json not found at {CONFIG_FILE}")
    with CONFIG_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


def _collect_known_acronyms(config: Dict) -> Dict[str, Set[str]]:
    """Return mapping: category_key → set of known uppercase acronyms."""
    known: Dict[str, Set[str]] = {}

    def _walk(node: Dict) -> None:
        for key, value in node.items():
            if key.startswith("_comment") or not isinstance(value, dict):
                continue
            cat = value.get("category")
            if cat:
                known.setdefault(cat, set()).add(key.upper())
            else:
                _walk(value)

    _walk(config.get("algorithms", {}))
    return known


def _parse_acronyms(raw: Optional[str]) -> Optional[List[str]]:
    """Parse a comma/space-separated acronym string into a list, or None."""
    if raw is None:
        return None
    parts = [p.strip().upper() for token in raw.split(",") for p in token.split() if p.strip()]
    return parts if parts else []


def _validate_all(
    selections: Dict[str, Optional[List[str]]],
    known_by_cat: Dict[str, Set[str]],
) -> None:
    errors: List[str] = []
    for cat, requested in selections.items():
        if not requested:
            continue
        unknown = set(requested) - known_by_cat.get(cat, set())
        if unknown:
            valid = sorted(known_by_cat.get(cat, set()))
            errors.append(
                f"  Category '{cat}': unknown acronym(s) {sorted(unknown)}.\n"
                f"    Valid values: {valid}"
            )
    if errors:
        _die("Validation failed:\n" + "\n".join(errors))


def _validate_features(drop_features: List[str], config: Dict) -> None:
    known = {k for k in config.get("optional_features", {}) if not k.startswith("_")}
    unknown = set(drop_features) - known
    if unknown:
        _die(
            f"Unknown optional features: {sorted(unknown)}. "
            f"Known features: {sorted(known)}"
        )


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    _log(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=check)


def _run_capture(cmd: List[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _current_branch() -> str:
    return _run_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def _create_branch(branch_name: str) -> None:
    _run(["git", "checkout", "-b", branch_name])


def _write_workflow_config(
    selections: Dict[str, Optional[List[str]]],
    drop_features: List[str],
    envs: Optional[List[str]],
    sim_datasets: Optional[List[str]],
    distributions: Optional[List[str]],
    network: Optional[List[str]],
    skip_build: bool,
    dry_run: bool,
) -> None:
    """Write ci/workflow_config.json with the arguments used for this export run.

    The file is committed onto the algo-export branch so package-and-build.yml
    can read the exact parameters without requiring manual workflow_dispatch input.
    """
    def _join(lst: Optional[List[str]]) -> str:
        return ",".join(lst) if lst else ""

    cfg: Dict[str, Any] = {
        "_comment": "Auto-generated by ci/packager.py. Do not edit manually.",
        "constructors":       _join(selections.get("constructor")),
        "selectors":          _join(selections.get("selector")),
        "improvement":        _join(selections.get("improvement")),
        "acceptance":         _join(selections.get("acceptance")),
        "joint":              _join(selections.get("joint")),
        "models":             _join(selections.get("model")),
        "rl_algorithms":      _join(selections.get("rl_algorithm")),
        "imitation_policies": _join(selections.get("imitation_policy")),
        "drop_features":      ",".join(drop_features),
        "envs":               _join(envs),
        "sim_datasets":       _join(sim_datasets),
        "distributions":      _join(distributions),
        "network":            _join(network),
        "skip_build":         skip_build,
        "dry_run":            dry_run,
    }
    out = CONFIG_FILE.parent / "workflow_config.json"
    out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    _log(f"Wrote {out.relative_to(PROJECT_ROOT)}")


def _delete_branch(branch_name: str, original_branch: str) -> None:
    _log(f"Rolling back: checking out '{original_branch}' and deleting '{branch_name}' …")
    subprocess.run(["git", "checkout", "--", "."], cwd=PROJECT_ROOT, check=False)
    subprocess.run(["git", "checkout", original_branch], cwd=PROJECT_ROOT, check=False)
    subprocess.run(["git", "branch", "-D", branch_name], cwd=PROJECT_ROOT, check=False)


def _commit_pruned_state(branch_name: str, selections: Dict[str, Optional[List[str]]]) -> None:
    _run(["git", "add", "-A"])
    summary_parts = [
        f"{cat}=[{', '.join(v)}]"
        for cat, v in selections.items()
        if v is not None
    ]
    summary = "; ".join(summary_parts) if summary_parts else "all categories"
    msg = f"algo-export: prune codebase — keep {summary}"
    result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PROJECT_ROOT)
    if result.returncode == 0:
        _log("No changes to commit after pruning (nothing was pruned).")
        return

    # First commit attempt — pre-commit hooks may auto-fix files (e.g. trailing whitespace).
    # If they do, the commit fails but the fixes are applied; re-stage and retry once.
    commit_result = subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=PROJECT_ROOT,
    )
    if commit_result.returncode != 0:
        _log("Pre-commit hook modified files — re-staging and retrying commit …")
        _run(["git", "add", "-A"])
        _run(["git", "commit", "-m", msg])


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_prune(
    selections: Dict[str, Optional[List[str]]],
    drop_features: List[str],
    dry_run: bool,
    envs: Optional[List[str]] = None,
    sim_datasets: Optional[List[str]] = None,
    distributions: Optional[List[str]] = None,
    network: Optional[List[str]] = None,
) -> None:
    if not PRUNE_SCRIPT.exists():
        _die(f"Prune script not found: {PRUNE_SCRIPT}")

    cmd: List[str] = [sys.executable, str(PRUNE_SCRIPT), "--config", str(CONFIG_FILE)]

    flag_map = {
        "constructor": "--constructors",
        "selector": "--selectors",
        "improvement": "--improvement",
        "acceptance": "--acceptance",
        "joint": "--joint",
        "model": "--models",
        "rl_algorithm": "--rl-algorithms",
        "imitation_policy": "--imitation-policies",
    }

    for cat, keep_list in selections.items():
        flag = flag_map[cat]
        if keep_list is None:
            continue
        cmd.append(flag)
        cmd.extend(keep_list)

    if drop_features:
        cmd.append("--drop-features")
        cmd.extend(drop_features)

    if envs is not None:
        cmd.append("--envs")
        cmd.extend(envs)

    if sim_datasets is not None:
        cmd.append("--sim-datasets")
        cmd.extend(sim_datasets)

    if distributions is not None:
        cmd.append("--distributions")
        cmd.extend(distributions)

    if network is not None:
        cmd.append("--network")
        cmd.extend(network)

    if dry_run:
        cmd.append("--dry-run")

    _run(cmd)


def _stage_profile_tool(includes: Optional[List[str]] = None) -> Path:
    """Copy the profile tool, config, patches and include files to a folder that survives pruning."""
    staged = Path(tempfile.mkdtemp(prefix="wsr-profile-"))
    config = json.loads(CONFIG_FILE.read_text())
    wanted = set(includes or [])
    for prof in config.get("export_profiles", {}).values():
        if isinstance(prof, dict):
            wanted.update(prof.get("include", []))
    for rel in sorted(wanted):
        src = PROJECT_ROOT / rel
        if src.is_symlink() or not src.exists():
            _die(f"--include path missing or a symlink: {rel}")
        dst = staged / "include" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    shutil.copy2(PROFILE_SCRIPT, staged / PROFILE_SCRIPT.name)
    shutil.copy2(CONFIG_FILE, staged / CONFIG_FILE.name)
    patches = CONFIG_FILE.parent / "export_profiles"
    if patches.is_dir():
        shutil.copytree(patches, staged / "export_profiles")
    return staged


def _step_profile(
    staged: Path,
    profile: Optional[str],
    entrypoints: Optional[str],
    engines: Optional[str],
    file_list: Optional[str],
    includes: List[str],
    dry_run: bool,
) -> None:
    """Apply the export profile, entry-point and engine choices to the pruned tree."""
    cmd = [sys.executable, str(staged / PROFILE_SCRIPT.name), "--root", str(PROJECT_ROOT),
           "--config", str(staged / CONFIG_FILE.name)]
    if profile:
        cmd += ["--profile", profile]
    if entrypoints:
        cmd += ["--entrypoints", entrypoints]
    if engines:
        cmd += ["--engines", engines]
    if file_list:
        cmd += ["--file-list", file_list]
    for rel in includes:
        cmd += ["--include", rel]
    cmd += ["--include-from", str(staged / "include")]
    if dry_run:
        cmd.append("--dry-run")
    _run(cmd)
    shutil.rmtree(staged, ignore_errors=True)


def _step_tree(output_file: Path) -> None:
    try:
        result = subprocess.run(
            ["tree", str(LOGIC_DIR), "--charset", "ascii", "--noreport", "-I", "__pycache__"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        output_file.write_text(result.stdout, encoding="utf-8")
        _log(f"Tree listing written to {output_file.relative_to(PROJECT_ROOT)}")
    except FileNotFoundError:
        _log("'tree' binary not found — falling back to Python directory walk.")
        lines: List[str] = []
        for p in sorted(LOGIC_DIR.rglob("*")):
            if any(part in _ZIP_EXCLUDE_DIRS for part in p.parts):
                continue
            if "__pycache__" in p.parts:
                continue
            indent = "  " * (len(p.relative_to(LOGIC_DIR).parts) - 1)
            lines.append(f"{indent}{p.name}{'/' if p.is_dir() else ''}")
        output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _log(f"Fallback listing written to {output_file.relative_to(PROJECT_ROOT)}")


def _step_build(dry_run: bool) -> Path:
    if not SPEC_FILE.exists():
        _die(f"simulator.spec not found at {SPEC_FILE}")

    dist_dir = PROJECT_ROOT / "dist"

    if dry_run:
        _log(f"[DRY-RUN] Would run: pyinstaller {SPEC_FILE} --clean")
        _log(f"[DRY-RUN] Build output would be at: {dist_dir}")
        return dist_dir

    _run([sys.executable, "-m", "PyInstaller", str(SPEC_FILE), "--clean"])

    if not dist_dir.exists():
        _die(f"PyInstaller ran but dist/ directory not found at {dist_dir}")

    _log(f"Build complete. Artifacts in: {dist_dir}")
    return dist_dir


def _step_zip(
    timestamp: str,
    dist_dir: Path,
    tree_file: Path,
    output_dir: Path,
    dry_run: bool,
) -> Path:
    zip_name = f"wsmart_route_export_{timestamp}.zip"
    zip_path = output_dir / zip_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        _log(f"[DRY-RUN] Would create zip at: {zip_path}")
        return zip_path

    def _should_exclude(path: Path) -> bool:
        for part in path.parts:
            if part in _ZIP_EXCLUDE_DIRS:
                return True
        return path.suffix in _ZIP_EXCLUDE_SUFFIXES

    _log(f"Creating archive: {zip_path}")
    files_added = 0

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for src_file in sorted(PROJECT_ROOT.rglob("*")):
            if not src_file.is_file():
                continue
            rel = src_file.relative_to(PROJECT_ROOT)
            if _should_exclude(rel):
                continue
            if rel.parts and rel.parts[0] == "dist":
                continue
            try:
                src_file.relative_to(output_dir)
                continue
            except ValueError:
                pass
            zf.write(src_file, rel)
            files_added += 1

        if dist_dir.exists():
            for dist_file in sorted(dist_dir.rglob("*")):
                if dist_file.is_file():
                    rel = dist_file.relative_to(PROJECT_ROOT)
                    zf.write(dist_file, rel)
                    files_added += 1

        if tree_file.exists():
            zf.write(tree_file, tree_file.relative_to(PROJECT_ROOT))
            files_added += 1

    _log(f"Archive complete: {zip_path} ({files_added} files, {zip_path.stat().st_size // 1024} KB)")
    return zip_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ci/packager.py",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--constructors", metavar="ACRONYMS", default=None)
    p.add_argument("--selectors", metavar="ACRONYMS", default=None)
    p.add_argument("--improvement", metavar="ACRONYMS", default=None)
    p.add_argument("--acceptance", metavar="ACRONYMS", default=None)
    p.add_argument("--joint", metavar="ACRONYMS", default=None)
    p.add_argument("--models", metavar="ACRONYMS", default=None)
    p.add_argument("--rl-algorithms", metavar="ACRONYMS", default=None, dest="rl_algorithms")
    p.add_argument("--imitation-policies", metavar="ACRONYMS", default=None, dest="imitation_policies")
    p.add_argument(
        "--drop-features",
        metavar="FEATURES",
        default=None,
        dest="drop_features",
        help=(
            "Comma/space-separated optional features to REMOVE "
            "(e.g. 'META_LEARNING,HPO,SECURITY,TRACKING'). "
            "Available: META_LEARNING, HPO, EVAL, SECURITY, CALLBACKS, ENUMS, DATA_WEB, TRACKING, CLI."
        ),
    )
    p.add_argument(
        "--envs",
        metavar="ENVS",
        default=None,
        help="Comma/space-separated environment names to KEEP (e.g. 'ptp'). Omit to keep all.",
    )
    p.add_argument(
        "--sim-datasets",
        metavar="NAMES",
        default=None,
        dest="sim_datasets",
        help="Comma/space-separated simulation dataset stems to KEEP (e.g. 'gen_dataset,sim_dataset').",
    )
    p.add_argument(
        "--distributions",
        metavar="NAMES",
        default=None,
        help="Comma/space-separated distribution stems to KEEP (e.g. 'gamma,empirical').",
    )
    p.add_argument(
        "--network",
        metavar="NAMES",
        default=None,
        help="Comma/space-separated network strategy stems to KEEP (e.g. 'file').",
    )
    p.add_argument(
        "--profile",
        metavar="NAME",
        default=None,
        help="Export profile from ci/export_config.json (export_profiles) applied after pruning.",
    )
    p.add_argument(
        "--entrypoints",
        choices=["both", "main", "dunder", "none"],
        default=None,
        help="Keep main.py, __main__.py, both (default) or neither in the package.",
    )
    p.add_argument(
        "--engines",
        metavar="SPEC",
        default=None,
        help="Solver frameworks per kept policy, e.g. 'swc_tcf=gurobi'. Default: all (or the profile's choice).",
    )
    p.add_argument(
        "--file-list",
        metavar="PATH",
        default=None,
        dest="file_list",
        help="Create source/ (main.py/__main__.py + logic/) and save the tree listing run inside it to PATH "
        "(relative to the package root, e.g. assets/files/FILE_LIST.txt).",
    )
    p.add_argument(
        "--include",
        metavar="PATH",
        action="append",
        default=[],
        help="Last step: copy PATH (file or directory, relative to the repository) into the package at the same "
        "location; repeatable. Added to the profile's include list.",
    )
    p.add_argument(
        "--output-dir", metavar="PATH",
        default=str(PROJECT_ROOT / "dist" / "exports"),
        dest="output_dir",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-build", action="store_true", dest="skip_build")
    p.add_argument("--no-commit", action="store_true", dest="no_commit")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_name = f"algo-export-{timestamp}"
    output_dir = Path(args.output_dir)
    tree_file = PROJECT_ROOT / "retained_logic_files.txt"

    # 1. Load and validate
    _log("Loading ci/export_config.json …")
    config = _load_config()
    known_by_cat = _collect_known_acronyms(config)

    selections: Dict[str, Optional[List[str]]] = {
        "constructor":       _parse_acronyms(args.constructors),
        "selector":          _parse_acronyms(args.selectors),
        "improvement":       _parse_acronyms(args.improvement),
        "acceptance":        _parse_acronyms(args.acceptance),
        "joint":             _parse_acronyms(args.joint),
        "model":             _parse_acronyms(args.models),
        "rl_algorithm":      _parse_acronyms(args.rl_algorithms),
        "imitation_policy":  _parse_acronyms(args.imitation_policies),
    }

    drop_features: List[str] = []
    if args.drop_features:
        drop_features = [
            f.strip().upper()
            for token in args.drop_features.split(",")
            for f in token.split()
            if f.strip()
        ]

    def _parse_list(raw: Optional[str]) -> Optional[List[str]]:
        if raw is None:
            return None
        parts = [p.strip() for token in raw.split(",") for p in token.split() if p.strip()]
        return parts  # empty list means "remove all"

    envs = _parse_list(args.envs)
    sim_datasets = _parse_list(args.sim_datasets)
    distributions = _parse_list(args.distributions)
    network = _parse_list(args.network)

    _validate_all(selections, known_by_cat)
    if drop_features:
        _validate_features(drop_features, config)

    if all(v is None for v in selections.values()):
        _log("WARNING: No category flags provided — all algorithms will be retained.")

    if args.dry_run:
        _log("=== DRY RUN MODE — no git or filesystem changes will be made ===")

    # 2. Record original branch
    original_branch = _current_branch()
    _log(f"Current branch: '{original_branch}'")
    _log(f"Export branch will be: '{branch_name}'")

    # 3. Create export branch
    if not args.dry_run:
        _create_branch(branch_name)
        _log(f"Checked out new branch '{branch_name}'.")
    else:
        _log(f"[DRY-RUN] Would create branch '{branch_name}'.")

    zip_path: Optional[Path] = None
    try:
        # 3b. Write workflow_config.json so it is captured in the commit
        if not args.dry_run:
            _write_workflow_config(
                selections, drop_features, envs, sim_datasets, distributions,
                network, args.skip_build, args.dry_run,
            )
        else:
            _log("[DRY-RUN] Would write ci/workflow_config.json.")

        # 4. Prune codebase (algorithms + optional features + subnets)
        _log("Step 1/4 — Pruning codebase …")
        # The pruner deletes logic/package/ when it finishes, so stage the profile tool first.
        profile_dir = (
            _stage_profile_tool(args.include)
            if (args.profile or args.entrypoints or args.engines or args.file_list or args.include)
            else None
        )
        _step_prune(
            selections,
            drop_features,
            dry_run=args.dry_run,
            envs=envs,
            sim_datasets=sim_datasets,
            distributions=distributions,
            network=network,
        )
        if profile_dir is not None:
            _step_profile(
                profile_dir, args.profile, args.entrypoints, args.engines, args.file_list, args.include, args.dry_run
            )

        # 5. Commit pruned state
        if not args.dry_run and not args.no_commit:
            _log("Committing pruned state …")
            _commit_pruned_state(branch_name, selections)

        # 6. Tree listing
        _log("Step 2/4 — Writing retained file tree …")
        _step_tree(tree_file)

        # 7. Build executable
        dist_dir: Path
        if args.skip_build:
            _log("Step 3/4 — Build step skipped (--skip-build).")
            dist_dir = PROJECT_ROOT / "dist" / "_nonexistent_skip_build"
        else:
            _log("Step 3/4 — Building executable with PyInstaller …")
            dist_dir = _step_build(dry_run=args.dry_run)

        # 8. Package into zip
        _log("Step 4/4 — Creating zip archive …")
        zip_path = _step_zip(timestamp, dist_dir, tree_file, output_dir, dry_run=args.dry_run)

        _log("=" * 60)
        _log("Export complete.")
        _log(f"  Branch  : {branch_name}")
        _log(f"  Archive : {zip_path}")
        _log(f"  Tree    : {tree_file}")
        _log("=" * 60)

    except Exception as exc:  # noqa: BLE001
        _log(f"Pipeline failed: {exc}")
        if not args.dry_run:
            _delete_branch(branch_name, original_branch)
        _die(f"Aborted. Rolled back to '{original_branch}'.")


if __name__ == "__main__":
    main()
