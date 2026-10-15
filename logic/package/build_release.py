"""Build the WSmart-Route release package from ``logic/release.yaml``.

The manifest is the single source of truth for the released package (owner
ruling Q-10a): registry keys per stage, released problems, and the
infrastructure seeds whose import closure ships. Everything else under ``logic/``
stays in ``main`` and is pruned from the package (Q-8/Q-8b).

Usage::

    # CI gate: every manifest key is registered; seeds and configs resolve.
    python logic/package/build_release.py --check

    # Build the release tree into <dir> and verify its import closure.
    python logic/package/build_release.py --build <dir>

    # Torch-free classical smoke test against a built tree (fresh
    # interpreter; CI runs it in a venv without torch installed).
    python logic/package/build_release.py --smoke <dir>

The build works on a staged copy; the working tree is never modified. Unreleased
code is not migrated or changed here — it simply does not ship (Q-8b migration
belongs to the config and simulator work packages).

Retired with this tool (Q-10a): ``ci/packager.py``, ``ci/export_config.json``,
``logic/package/prune_codebase.py`` and the ``remove_*.py`` scripts, and the
``ci/export_profiles/`` patch flow. The manifest uses registry keys, not the old
BB/BC/BPC acronyms; the Studio app keeps its own display-name map.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[2]
LOGIC = REPO / "logic"

STAGES: Dict[str, Path] = {
    "routing": LOGIC / "src" / "policies" / "routing" / "_manifest.py",
    "packing": LOGIC / "src" / "policies" / "packing" / "_manifest.py",
    "route_improvement": LOGIC / "src" / "policies" / "route_improvement" / "_manifest.py",
    "acceptance": LOGIC / "src" / "policies" / "acceptance_criteria" / "_manifest.py",
    "packing_and_routing": LOGIC / "src" / "policies" / "packing_and_routing" / "_manifest.py",
}

# logic/ subtrees that ship in the release. Anything else (benchmark, cloud,
# docs, gen, package, validation) is development tooling and stays out (Q-11b).
STAGED_TOPS = ("src", "configs", "controllers")

ENTRY_RE = re.compile(r"^\s*'([A-Za-z0-9_-]+)':\s*'(logic\.[A-Za-z0-9_.:]+)'\s*,?\s*$")
LAZY_LINE_RE = re.compile(r'^(\s*)"([A-Za-z_][A-Za-z0-9_]*)":\s*"(logic\.[A-Za-z0-9_.:]+)"\s*,?\s*$', re.MULTILINE)
LOGIC_MODULE_RE = re.compile(r"\blogic(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
DECLARE_RE = re.compile(r"\.declare\('([A-Za-z0-9_-]+)',\s*ENTRIES\['([A-Za-z0-9_-]+)']\)")


# --------------------------------------------------------------------------- #
# Manifest loading                                                            #
# --------------------------------------------------------------------------- #


def load_manifest() -> dict:
    """Parse ``logic/release.yaml`` and return its plain-dict form."""
    import yaml

    with (LOGIC / "release.yaml").open() as fh:
        manifest = yaml.safe_load(fh)
    for section in ("version", "problems", "stages", "infrastructure"):
        if section not in manifest:
            raise SystemExit(f"release.yaml: missing required section {section!r}")
    manifest.setdefault("excluded_packages", [])
    return manifest


def parse_registry_manifest(path: Path) -> Dict[str, str]:
    """Return {registry key: 'module:Class'} from a generated ``_manifest.py``."""
    entries: Dict[str, str] = {}
    in_entries = False
    for line in path.read_text().splitlines():
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


# --------------------------------------------------------------------------- #
# Module/file mapping (works for the repo tree or a staged build tree)        #
# --------------------------------------------------------------------------- #


def module_to_file(module: str, root: Path) -> Optional[Path]:
    """Map a dotted module under ``<root>/logic`` to its file, or None."""
    candidate = root.joinpath(*module.split("."))
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    return None


def file_to_module(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def trimmed(module: str, root: Path) -> str:
    """Longest prefix of ``module`` that maps to a file under ``root``."""
    parts = module.split(".")
    while parts:
        candidate = ".".join(parts)
        if module_to_file(candidate, root) is not None:
            return candidate
        parts = parts[:-1]
    return ""


# --------------------------------------------------------------------------- #
# Import analysis                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class Imports:
    module_level: Set[str] = field(default_factory=set)
    deferred: Set[str] = field(default_factory=set)


def _package_of(path: Path, root: Path) -> List[str]:
    """Dotted package path containing ``path`` (the module's own name stripped)."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    parts = parts[:-1]  # works for both module files and __init__.py
    return parts


def imports_of(path: Path, root: Path) -> Imports:
    """Collect logic.* imports of a file, split by module level vs deferred."""
    result = Imports()
    tree = ast.parse(path.read_text())
    pkg = _package_of(path, root)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = (node.module or "").split(".")
                names = [".".join(base + [alias.name]) for alias in node.names]
                if node.module:
                    names.append(node.module)
            else:
                up = pkg[: len(pkg) - (node.level - 1)]
                base = up + ([node.module] if node.module else [])
                names = [".".join(base + [alias.name]) for alias in node.names]
                names.append(".".join(base))
        else:
            continue
        for name in names:
            if not name.startswith("logic"):
                continue
            name = name.rstrip(".")
            if name and _is_deferred(tree, node):
                result.deferred.add(name)
            else:
                result.module_level.add(name)
    return result


def _is_deferred(tree: ast.Module, target: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if inner is target:
                    return True
    return False


# --------------------------------------------------------------------------- #
# Closure computation                                                         #
# --------------------------------------------------------------------------- #


def compute_closure(manifest: dict) -> Tuple[Set[Path], List[str]]:  # noqa: C901
    """Keep the module-level import closure of the seeds minus exclusions."""
    excluded = tuple(manifest["excluded_packages"])
    stage_entries = {stage: parse_registry_manifest(path) for stage, path in STAGES.items()}
    seeds: Set[str] = set(manifest["infrastructure"])
    for stage, keys in manifest["stages"].items():
        for key in keys:
            target = stage_entries[stage].get(key)
            if target is None:
                raise SystemExit(f"release.yaml: stage {stage!r} key {key!r} is not registered")
            seeds.add(target.split(":")[0])

    kept_modules: Set[str] = set()
    notes: List[str] = []

    def enqueue(module: str) -> None:
        """Add a module and all its ancestor packages to the work queue."""
        parts = module.split(".")
        for depth in range(len(parts), 0, -1):
            candidate = ".".join(parts[:depth])
            if module_to_file(candidate, REPO) is None:
                continue
            if candidate not in kept_modules and not candidate.startswith(excluded):
                queue.append(candidate)
            if candidate in kept_modules:
                continue

    # name -> target module, harvested from _LAZY maps in package inits:
    # resolves "from <pkg> import <Name>" when the name lives in a submodule.
    lazy_index: Dict[str, Dict[str, str]] = {}
    # Modules discovered through a package's public surface (lazy map or
    # deferred re-export). If such a module's chain hard-imports an excluded
    # package, the module is unreleasable and is dropped from the closure:
    # the code path is simply unavailable in the release (Q-6/Q-8).
    surface: Set[str] = set()

    queue: List[str] = list(seeds)
    while queue:
        module = queue.pop()
        root = trimmed(module, REPO)
        if not root or root in kept_modules:
            continue
        if root.startswith(excluded):
            continue
        kept_modules.add(root)
        # The module's ancestor packages ship too (skeleton), and their eager
        # imports are part of the closure.
        enqueue(root)
        path = module_to_file(root, REPO)
        if path is None:  # pragma: no cover - trimmed() guarantees a file
            continue
        imports = imports_of(path, REPO)
        # Package re-export maps and deferred imports of kept __init__ files
        # are part of the package's public surface. Deferred imports are the
        # runtime opt-in pattern: an excluded target is skipped with a note
        # (the code path is simply unavailable in the release).
        deferred: Set[str] = set(imports.deferred)
        pending: List[str] = sorted(imports.module_level | deferred)
        if path.name == "__init__.py":
            # A lazy map is indexed, not followed: a target ships only when a kept
            # module imports its name (below). Following every entry would ship
            # every architecture and algorithm behind the facades (Q-12).
            for match in LAZY_LINE_RE.finditer(path.read_text()):
                pkg_name, target = match.group(2), match.group(3).split(":")[0]
                lazy_index.setdefault(root, {})[pkg_name] = target
            surface.update(trimmed(dep, REPO) for dep in imports.deferred)
        for dep in pending:
            root_dep = trimmed(dep, REPO)
            if not root_dep:
                notes.append(f"unresolved module-level import in {root}: {dep}")
                continue
            # "from <pkg> import <Name>" where Name is served by the
            # package's lazy map: enqueue the real target module.
            if module_to_file(dep, REPO) is None and dep != root_dep:
                name = dep[len(root_dep) + 1 :]
                if "." not in name:
                    target = lazy_index.get(root_dep, {}).get(name)
                    if target and target not in kept_modules:
                        surface.add(trimmed(target, REPO))
                        queue.append(target)
            if root_dep.startswith(excluded):
                if path.name != "__init__.py" and dep not in deferred:
                    if root in surface:
                        kept_modules.discard(root)
                        notes.append(
                            f"surface module {root} dropped: its chain module-level "
                            f"imports excluded {root_dep} (path unavailable in the release)"
                        )
                        break
                    raise SystemExit(
                        f"kept file {path} imports excluded package {root_dep!r} at module "
                        "level; move the import into a function, or extend the manifest"
                    )
                notes.append(f"excluded {root_dep} stays out of the release ({path.name})")
                continue
            if root_dep not in kept_modules:
                queue.append(root_dep)

    files: Set[Path] = set()
    for module in kept_modules:
        path = module_to_file(module, REPO)
        if path is not None:
            files.add(path)
    # The _manifest.py files always ship (filtered to released keys).
    for stage_path in STAGES.values():
        files.add(stage_path)
    return files, notes


FROM_IMPORT_RE = re.compile(r"^\s*from (logic(?:\.\w+)+) import \(?([\w ,\n]+)\)?", re.M)


def release_tests(manifest: dict, kept: Set[Path]) -> Tuple[List[Path], List[Path]]:
    """Split logic/test files into release tests and pruned tests (P-06)."""
    release: List[Path] = []
    pruned: List[Path] = []
    kept_modules = {file_to_module(f, REPO) for f in kept}
    for path in sorted((LOGIC / "test").rglob("*.py")):
        modules = set()
        text = path.read_text()
        for match in LOGIC_MODULE_RE.finditer(text):
            root = trimmed(match.group(0), REPO)
            if root:
                modules.add(root)
        # ``from <pkg> import <submodule>`` names a module the regex above sees as <pkg>.
        for match in FROM_IMPORT_RE.finditer(text):
            for name in re.findall(r"\w+", match.group(2)):
                if module_to_file(f"{match.group(1)}.{name}", REPO) is not None:
                    modules.add(f"{match.group(1)}.{name}")
        if modules <= kept_modules:
            release.append(path)
        else:
            pruned.append(path)
    return release, pruned


# --------------------------------------------------------------------------- #
# Staging transforms                                                           #
# --------------------------------------------------------------------------- #


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
    """Drop _LAZY entries whose target module is pruned."""
    dropped: List[str] = []
    lines = path.read_text().splitlines(keepends=True)
    kept_lines: List[str] = []
    for line in lines:
        lazy = LAZY_LINE_RE.match(line.rstrip("\n"))
        if lazy:
            target = lazy.group(3).split(":")[0]
            if module_to_file(target, REPO) is not None:
                ok = target in staged_modules
            else:
                ok = trimmed(target, out) in staged_modules
            if not ok:
                dropped.append(f"{path.relative_to(out)}: lazy {lazy.group(2)} -> {target}")
                continue
        kept_lines.append(line)
    path.write_text("".join(kept_lines))
    return dropped


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


# --------------------------------------------------------------------------- #
# Check / build / smoke                                                        #
# --------------------------------------------------------------------------- #


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


def run_check(manifest: dict) -> int:  # noqa: C901
    problems: List[str] = []
    stage_entries = {stage: parse_registry_manifest(path) for stage, path in STAGES.items()}
    for stage, keys in manifest["stages"].items():
        for key in keys:
            if key not in stage_entries[stage]:
                problems.append(f"stage {stage}: key {key!r} is not registered in _manifest.py")
    for problem_name in manifest["problems"]:
        if not (LOGIC / "src" / "envs" / "tasks" / f"{problem_name}.py").is_file():
            problems.append(f"problem {problem_name!r}: no envs/tasks/{problem_name}.py")
    for seed in manifest["infrastructure"]:
        if module_to_file(seed, REPO) is None:
            problems.append(f"infrastructure seed {seed!r} does not resolve to a file")
    for excluded in manifest["excluded_packages"]:
        if module_to_file(excluded, REPO) is None:
            problems.append(f"excluded package {excluded!r} does not resolve to a file")
    routing = set(manifest["stages"].get("routing", ()))
    packing = set(manifest["stages"].get("packing", ()))
    for key in routing:
        if not (LOGIC / "configs" / "policies" / f"policy_{key}.yaml").is_file():
            problems.append(f"routing key {key!r}: no configs/policies/policy_{key}.yaml")
    for key in packing:
        if not (LOGIC / "configs" / "policies" / "other" / f"pk_{key}.yaml").is_file():
            problems.append(f"packing key {key!r}: no configs/policies/other/pk_{key}.yaml")
    for stage, keys in manifest["stages"].items():
        for key in keys:
            target = stage_entries[stage].get(key, "")
            if target and module_to_file(target.split(":")[0], REPO) is None:
                problems.append(f"stage {stage} key {key!r}: target module missing ({target})")
    if problems:
        print("release manifest check FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    total = sum(len(keys) for keys in manifest["stages"].values())
    print(
        f"release manifest check OK: {total} stage keys, {len(manifest['problems'])} problems, "
        f"{len(manifest['infrastructure'])} infrastructure seeds"
    )
    return 0


def run_build(manifest: dict, out: Path) -> int:  # noqa: C901
    out = out.resolve()
    if out in (REPO, LOGIC) or out in REPO.parents or LOGIC in out.parents:
        raise ValueError("Release destination must not contain the repository or lie inside logic/")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Release destination must be absent or empty; existing files are never deleted")
    kept, notes = compute_closure(manifest)

    # --- stage the copy -------------------------------------------------- #
    (out / "logic").mkdir(parents=True)
    for top in STAGED_TOPS:
        source = LOGIC / top
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(LOGIC)
            if path.suffix == ".py" and path not in kept:
                continue
            dest = out / "logic" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    shutil.copy2(LOGIC / "release.yaml", out / "logic" / "release.yaml")
    # logic must be a regular package in the staging tree; without __init__.py
    # it degrades to a namespace package and loses to any other installed copy.
    shutil.copy2(LOGIC / "__init__.py", out / "logic" / "__init__.py")

    # --- the bin-analysis package (RQ-grok-7 a): runtime modules only ------ #
    bin_analysis = REPO / "packages" / "wsmart_bin_analysis"
    if not (bin_analysis / "__init__.py").is_file():
        raise SystemExit(f"{bin_analysis} is empty: run git submodule update --init packages/wsmart_bin_analysis")
    for path in [
        bin_analysis / "__init__.py",
        *sorted((bin_analysis / "export").rglob("*.py")),
        *sorted((bin_analysis / "python").rglob("*.py")),
    ]:
        if "__pycache__" in path.parts or "notebooks" in path.parts:
            continue
        dest = out / "packages" / "wsmart_bin_analysis" / path.relative_to(bin_analysis)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    (out / "packages" / "__init__.py").write_text('"""Vendored companion packages of the release."""\n')

    # --- registry manifests: released keys only -------------------------- #
    for stage, stage_path in STAGES.items():
        rel = stage_path.relative_to(LOGIC)
        staged_manifest = out / "logic" / rel
        if staged_manifest.is_file():
            notes += filter_manifest_file(staged_manifest, set(manifest["stages"].get(stage, ())))

    # --- package configs: released YAML only ----------------------------- #
    routing = set(manifest["stages"].get("routing", ()))
    packing = set(manifest["stages"].get("packing", ()))
    for yaml_path in sorted((out / "logic" / "configs" / "policies").glob("policy_*.yaml")):
        if yaml_path.stem.removeprefix("policy_") not in routing:
            yaml_path.unlink()
    other_dir = out / "logic" / "configs" / "policies" / "other"
    if other_dir.is_dir():
        for yaml_path in sorted(other_dir.glob("*.yaml")):
            if yaml_path.stem.removeprefix("pk_") not in packing:
                yaml_path.unlink()

    # --- tests: release CI set only (P-06) ------------------------------- #
    release, pruned = release_tests(manifest, kept)
    test_out = out / "logic" / "test"
    for path in release:
        rel = path.relative_to(LOGIC / "test")
        dest = test_out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    # --- lazy maps and eager init imports -------------------------------- #
    staged_modules = {file_to_module(p, out) for p in (out / "logic").rglob("*.py")}
    for staged in sorted((out / "logic").rglob("__init__.py")):
        notes += filter_lazy_map(staged, staged_modules, out)
        notes += drop_pruned_init_imports(staged, staged_modules, out)

    # --- release test selection: validated against the staged tree -------- #
    staged_tests = [test_out / p.relative_to(LOGIC / "test") for p in release]
    release, collect_failures = validate_release_tests(out, staged_tests)
    (test_out / "release_tests.txt").write_text("\n".join(str(p.relative_to(test_out)) for p in release) + "\n")
    if collect_failures:
        (test_out / "release_tests.excluded.txt").write_text(
            "# Failed collection against the staged tree (dynamically imported "
            "modules outside the release):\n" + "\n".join(sorted(collect_failures)) + "\n"
        )
        notes.append(f"{len(collect_failures)} test files excluded by collection validation")

    # --- release pyproject + docs index ---------------------------------- #
    write_release_pyproject(manifest, out)
    write_docs_index(manifest, out)

    # --- verification ----------------------------------------------------- #
    staged_modules = {file_to_module(p, out) for p in (out / "logic").rglob("*.py")}
    errors = verify_staged(out, staged_modules)
    staged_py = list((out / "logic").rglob("*.py"))
    lines = sum(len(p.read_text().splitlines()) for p in staged_py)
    print(
        f"built {out}: {len(staged_py)} python files, {lines} lines, "
        f"{len(release)} release test files (pruned {len(pruned)})"
    )
    for note in notes:
        print(f"  note: {note}")
    if errors:
        print("staged closure verification FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("staged closure verification OK: no module-level import escapes the manifest")
    return 0


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


def write_release_pyproject(manifest: dict, out: Path) -> None:
    """Generate the installable release pyproject from logic/pyproject.toml."""
    try:  # Python 3.11+
        import tomllib as toml_reader
    except ModuleNotFoundError:  # Python 3.10
        import tomli as toml_reader

    base = toml_reader.loads((LOGIC / "pyproject.toml").read_text())
    project = base.get("project", {})
    deps = project.get("dependencies", [])
    neural = project.get("optional-dependencies", {}).get("neural", [])
    if not neural:
        raise SystemExit("logic/pyproject.toml: [project.optional-dependencies].neural is missing")
    lines = [
        "# Generated by logic/package/build_release.py from logic/release.yaml and",
        "# logic/pyproject.toml. Do not edit; regenerate the release instead.",
        "",
        "[build-system]",
        'requires = ["setuptools>=68"]',
        'build-backend = "setuptools.build_meta"',
        "",
        "[project]",
        f'name = "{project.get("name", "wsmart-route")}"',
        f'version = "{manifest["version"]}"',
    ]
    if project.get("description"):
        lines.append(f'description = "{project["description"]}"')
    if project.get("requires-python"):
        lines.append(f'requires-python = "{project["requires-python"]}"')
    # The vendored bin-analysis package plots with matplotlib on import.
    if not any(dep.split(">")[0].split("=")[0].strip() == "matplotlib" for dep in deps):
        deps = [*deps, "matplotlib>=3.9.4"]
    lines.append("dependencies = [")
    lines += [f'    "{dep}",' for dep in deps]
    lines.append("]")
    lines += [
        "",
        "[project.optional-dependencies]",
        "neural = [",
        *[f'    "{dep}",' for dep in neural],
        "]",
        "",
        "[tool.setuptools.packages.find]",
        'include = ["logic*", "packages*"]',
        "",
    ]
    (out / "pyproject.toml").write_text("\n".join(lines) + "\n")


def write_docs_index(manifest: dict, out: Path) -> None:
    """Generate the release docs index from the manifest (P-05: docs driver)."""
    stage_entries = {stage: parse_registry_manifest(path) for stage, path in STAGES.items()}
    lines = [
        "<!-- Generated by logic/package/build_release.py from logic/release.yaml. -->",
        f"# WSmart-Route {manifest['version']} release contents",
        "",
        "Problems: " + ", ".join(f"`{p}`" for p in manifest["problems"]) + ".",
        "",
        "| Stage | Registry key | Module |",
        "|---|---|---|",
    ]
    for stage, keys in manifest["stages"].items():
        for key in keys:
            target = stage_entries[stage].get(key, "")
            module = target.split(":")[0].replace("logic.src.policies.", "")
            lines.append(f"| {stage} | `{key}` | `{module}` |")
    lines += [
        "",
        "## Infrastructure",
        "",
        *[f"- `{seed}`" for seed in manifest["infrastructure"]],
        "",
        "The package installs without PyTorch; install `wsmart-route[neural]` for the "
        "training stack (Q-16a). Unreleased components stay in the development tree "
        "and are excluded from this package (Q-8).",
        "",
    ]
    docs_dir = out / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "release.md").write_text("\n".join(lines))


SMOKE_SCRIPT = r"""
import sys

sys.path.insert(0, sys.argv[1])
assert "torch" not in sys.modules

from logic.src.policies.packing.base.registry import PackingRegistry  # noqa: F401
from logic.src.policies.route_improvement.base.registry import RouteImproverRegistry  # noqa: F401
from logic.src.policies.acceptance_criteria.base.registry import AcceptanceCriterionRegistry  # noqa: F401
from logic.src.utils.registry import ACCEPTANCE, IMPROVEMENT, PACKING

assert PACKING.get("last_minute") is not None
assert PACKING.get("service_level") is not None
assert PACKING.get("lookahead") is not None
assert PACKING.get("regular") is not None
assert IMPROVEMENT.get("classical_local_search") is not None
assert IMPROVEMENT.get("fast_tsp") is not None
assert ACCEPTANCE.get("bmc") is not None
assert ACCEPTANCE.get("oi") is not None

# Unreleased keys must not resolve (Q-8).
assert PACKING.get("revenue_threshold") is None, "unreleased packing key resolves"
assert ACCEPTANCE.get("lahc") is None, "unreleased acceptance key resolves"
assert IMPROVEMENT.get("lkh") is None, "unreleased improver key resolves"

PACKING.create("last_minute")
PACKING.create("regular")
ACCEPTANCE.create("bmc", initial_temp=100.0, alpha=0.995)
ACCEPTANCE.create("oi")
IMPROVEMENT.create("classical_local_search")
IMPROVEMENT.create("fast_tsp")

assert "torch" not in sys.modules, "torch was imported on the classical path"
print("smoke OK: classical components resolve without torch")

# The released routing algorithms, the simulator and its bin data import from the
# built tree. They still reach torch through base_routing_policy -> simulations.repository
# and the day-loop states (Q-16a is open), so this runs after the torch-free checks.
from logic.src.policies.routing.base.registry import RoutingAlgorithmRegistry  # noqa: F401
from logic.src.utils.registry import ROUTING

for key in ("swc_tcf", "bpc", "alns", "sans", "hgs", "psoma", "aco_hh", "pg_clns", "cvrp", "na"):
    assert ROUTING.get(key) is not None, f"released routing key {key} does not resolve"
assert ROUTING.get("hgs_alns") is None, "unreleased routing key resolves"
import logic.src.pipeline.simulations.simulator  # noqa: F401
from packages.wsmart_bin_analysis import GridBase  # noqa: F401

print("smoke OK: released routing keys and the simulator import from the release tree")
"""


def run_smoke(out: Path) -> int:
    out = out.resolve()
    if not (out / "logic").is_dir():
        raise SystemExit(f"{out} is not a built release tree (run --build first)")
    result = subprocess.run(
        [sys.executable, "-c", SMOKE_SCRIPT, str(out)],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="")
        print("smoke FAILED")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate the manifest against the tree")
    parser.add_argument("--build", metavar="DIR", help="build the release tree into DIR")
    parser.add_argument("--smoke", metavar="DIR", help="torch-free smoke test against a built tree")
    args = parser.parse_args()
    manifest = load_manifest()
    if args.check:
        return run_check(manifest)
    if args.build:
        return run_build(manifest, Path(args.build))
    if args.smoke:
        return run_smoke(Path(args.smoke))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
