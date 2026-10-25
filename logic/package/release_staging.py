"""Pruning of the staged release tree: registry manifests, lazy maps and package initialisers trimmed to the closure, plus the post-build checks.

Attributes:
    filter_manifest_file: Keep only the released keys of a registry manifest.
    filter_lazy_map: Drop lazy-map entries whose modules are not staged.
    drop_pruned_init_imports: Remove ``__init__`` imports of pruned modules.
    drop_unbound_exports: Remove ``__all__`` and docstring entries for names an ``__init__`` no longer binds.
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
from typing import List, Optional, Set, Tuple

from logic.package.release_closure import (
    ENTRY_RE,
    REPO,
    _guards_import,
    _is_deferred,
    _package_of,
    eager_targets,
    imports_of,
    lazy_targets,
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
    rewrite_spans: List[tuple] = []
    for node in tree.body:
        targets = _import_targets(node, path, root)
        if targets is None:
            continue
        if isinstance(node, ast.ImportFrom) and len(node.names) > 1:
            kept_aliases = [
                alias
                for alias, target in zip(node.names, targets, strict=False)
                if _import_resolves(target, staged_modules, out)
            ]
            if 0 < len(kept_aliases) < len(node.names):
                rewrite_spans.append((node.lineno, node.end_lineno, kept_aliases, node))
                dropped.extend(
                    f"{path.relative_to(out)}: import {target}"
                    for alias, target in zip(node.names, targets, strict=False)
                    if alias not in kept_aliases
                )
                continue
        for target in targets:
            if not target.startswith("logic"):
                continue
            if isinstance(node, ast.ImportFrom) and not _import_resolves(target, staged_modules, out):
                remove_spans.append((node.lineno, node.end_lineno))
                dropped.append(f"{path.relative_to(out)}: import {target}")
                break
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
    if remove_spans or rewrite_spans:
        lines = path.read_text().splitlines(keepends=True)
        edits = [(start, end, None, None) for start, end in remove_spans] + rewrite_spans
        for start, end, aliases, node in sorted(edits, key=lambda edit: edit[0], reverse=True):
            replacement = []
            if aliases is not None:
                stmt = ast.ImportFrom(module=node.module, names=aliases, level=node.level)
                replacement = [ast.unparse(stmt) + "\n"]
            lines[start - 1 : end] = replacement
        path.write_text("".join(lines))
        drop_unbound_exports(path)
        _check_no_dangling_names(path, tree)
    return dropped


def _check_no_dangling_names(path: Path, original: ast.Module) -> None:
    """Fail when pruning removed an import whose name the init still uses (a NameError at run time)."""
    pruned = ast.parse(path.read_text())
    lost = _bound_names(original) - _bound_names(pruned)
    used = {node.id for node in ast.walk(pruned) if isinstance(node, ast.Name)}
    if lost & used:
        raise SystemExit(f"{path}: pruned imports of {sorted(lost & used)}, which the module still uses")


def _import_targets(node: ast.stmt, path: Path, root: Path) -> Optional[List[str]]:
    """Dotted targets of an import statement (one per alias, then the from-module); None for other nodes."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return None
    if node.level == 0:
        base = (node.module or "").split(".")
        targets = [".".join(base + [alias.name]) for alias in node.names]
        if node.module:
            targets.append(node.module)
        return targets
    pkg = _package_of(path, root)
    up = pkg[: len(pkg) - (node.level - 1)]
    base = up + ([node.module] if node.module else [])
    return [".".join(base + [alias.name]) for alias in node.names] + [".".join(base)]


def _import_resolves(target: str, staged_modules: Set[str], out: Path) -> bool:
    """Whether ``from <pkg> import <name>`` (as ``<pkg>.<name>``) still resolves in the staged tree.

    The name is a staged submodule, or a name the staged package init binds (its own pruned
    imports already removed: inits are processed deepest first).
    """
    if not target.startswith("logic"):
        return True
    if target in staged_modules:
        return True
    if module_to_file(target, REPO) is not None:
        return False  # a real module that did not ship
    package, _, name = target.rpartition(".")
    if package not in staged_modules:
        return False
    init = out / Path(*package.split(".")) / "__init__.py"
    if not init.is_file():
        # ``from <module> import <name>``: the name must be bound in that module.
        module_file = (out / Path(*package.split("."))).with_suffix(".py")
        if not module_file.is_file():
            return False
        tree = ast.parse(module_file.read_text())
        return name == "*" or name in _bound_names(tree) or name in eager_targets(module_file, out)
    if name == "*" or name in _bound_names(ast.parse(init.read_text())) or name in eager_targets(init, out):
        return True
    lazy = lazy_targets(init).get(name)  # served by the package's lazy map
    return lazy is not None and trimmed(lazy, out) in staged_modules


ATTRIBUTE_LINE_RE = re.compile(r"^    ([A-Za-z_][A-Za-z0-9_]*): ")


def _bound_names(tree: ast.Module) -> Set[str]:
    """Names a module binds at top level (imports, definitions, assignments)."""
    bound: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            bound.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
    return bound


def _all_node(tree: ast.Module) -> Optional[ast.Assign]:
    """The module's literal ``__all__ = [...]`` assignment, if any."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            return node
    return None


def _rewrite_all(lines: List[str], node: ast.Assign, bound: Set[str]) -> List[str]:
    """``lines`` with the ``__all__`` assignment rewritten one name per line, keeping only bound names."""
    names = [e.value for e in node.value.elts if isinstance(e, ast.Constant) and e.value in bound]
    block = ["__all__ = [\n", *(f'    "{name}",\n' for name in names), "]\n"]
    return lines[: node.lineno - 1] + block + lines[node.end_lineno :]


def _unbound_all_lines(path: Path, tree: ast.Module, bound: Set[str]) -> Optional[Set[int]]:
    """Line indexes of ``__all__`` entries that name unbound names; None when the list must be rewritten."""
    node = _all_node(tree)
    lines: Set[int] = set()
    if node is None:
        return lines
    for element in node.value.elts:
        if isinstance(element, ast.Constant) and element.value not in bound:
            if element.lineno in (node.lineno, node.end_lineno) or element.lineno != element.end_lineno:
                return None  # entries share lines: rewrite the whole list
            lines.add(element.lineno - 1)
    return lines


def _unbound_attribute_lines(tree: ast.Module, lines: List[str], bound: Set[str]) -> Set[int]:
    """Line indexes of the module docstring's ``Attributes:`` entries for unbound names."""
    if not tree.body or ast.get_docstring(tree, clean=False) is None:
        return set()
    drop: Set[int] = set()
    in_attributes = False
    for index in range(tree.body[0].lineno - 1, tree.body[0].end_lineno):
        text = lines[index].rstrip("\n")
        if text.strip() == "Attributes:":
            in_attributes = True
        elif in_attributes and text.strip():
            match = ATTRIBUTE_LINE_RE.match(text)
            if match is None:
                in_attributes = False
            elif match.group(1) not in bound:
                drop.add(index)
    return drop


def drop_unbound_exports(path: Path) -> None:
    """Drop the ``__all__`` names, and the docstring ``Attributes:`` lines, of names the module no longer binds."""
    tree = ast.parse(path.read_text())
    lines = path.read_text().splitlines(keepends=True)
    bound = _bound_names(tree)
    all_lines = _unbound_all_lines(path, tree, bound)
    drop = (all_lines or set()) | _unbound_attribute_lines(tree, lines, bound)
    if drop:
        lines = [line if index not in drop else None for index, line in enumerate(lines)]
    if all_lines is None:
        node = _all_node(tree)
        lines = _rewrite_all([line if line is not None else "" for line in lines], node, bound)
    if drop or all_lines is None:
        path.write_text("".join(line for line in lines if line is not None))


def write_release_conftest(source: Path, out: Path) -> List[str]:
    """Stage ``logic/test/conftest.py`` with only the fixture plugins that ship; return the dropped ones.

    The source conftest registers every fixture module; the fixtures of excluded components
    (meta-RL, ...) are pruned with them, and pytest refuses to start on a missing plugin.
    """
    text = source.read_text()
    match = re.search(r"^pytest_plugins = \[\n(.*?)^\]", text, re.M | re.S)
    if match is None:
        raise SystemExit(f"{source}: no pytest_plugins list")
    dropped: List[str] = []
    kept_lines: List[str] = []
    for line in match.group(1).splitlines(keepends=True):
        name = line.strip().strip(",").strip('"')
        if name and not (out / Path(*name.split("."))).with_suffix(".py").is_file():
            dropped.append(name)
            continue
        kept_lines.append(line)
    dest = out / "logic" / "test" / "conftest.py"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text[: match.start(1)] + "".join(kept_lines) + text[match.end(1) :])
    return dropped


_IMPORT_SWEEP = r"""
import importlib, pathlib, sys, traceback
root = pathlib.Path.cwd()
failures = 0
for path in sorted((root / "logic").rglob("*.py")):
    rel = path.relative_to(root)
    if rel.parts[1] == "test" or "-" in str(rel):
        continue
    name = ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
    try:
        importlib.import_module(name)
    except Exception as exc:
        failures += 1
        print(f"IMPORT-FAIL {name}: {type(exc).__name__}: {exc}"[:300])
sys.exit(1 if failures else 0)
"""


def import_sweep(out: Path, python: Optional[str] = None) -> List[str]:
    """Import every shipped module of the staged tree with ``python``; return the failures.

    The closure check is static; a module-level import that resolves only through a lazy
    registry (``from logic.src.envs import CPTPTWTC`` when that problem's
    module is pruned) passes it and still breaks at import time in the release.
    """
    import os
    import shutil

    env = dict(os.environ, PYTHONPATH=str(out))
    result = subprocess.run(
        [python or sys.executable, "-c", _IMPORT_SWEEP], cwd=out, env=env, capture_output=True, text=True, timeout=3600
    )
    for cache in out.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    return [line.removeprefix("IMPORT-FAIL ") for line in result.stdout.splitlines() if line.startswith("IMPORT-FAIL ")]


_RESULT_RE = re.compile(r"^(?:FAILED|ERROR) (\S+?\.py)(?:::\S*)?(?: - (.*))?$")


def validate_release_tests(
    out: Path,
    staged_tests: List[Path],
    python: Optional[str] = None,
    data_dir: Optional[Path] = None,
    rounds: int = 5,
) -> Tuple[List[Path], List[str]]:
    """Keep only the release tests that pass when run in the staged tree.

    The static selection (imports inside the closure) cannot see dynamically constructed
    imports or tests that need excluded behaviour, so the staged copy is authoritative: the
    release tests run in it with ``python`` (the release environment; the build's own
    interpreter by default), and every file with a failing, erroring or uncollectable test is
    dropped. Rounds repeat until a run is clean. ``data_dir`` is linked in as ``data/`` for
    the run (the package's data ships separately) and removed afterwards, together with
    everything else the run created.

    Returns:
        The kept test paths, and one ``path: reason`` line per excluded file.
    """
    import os
    import shutil

    remaining = list(staged_tests)
    excluded: dict = {}
    before = {p.name for p in out.iterdir()}
    link = out / "data"
    linked = False
    if data_dir is not None and not link.exists():
        link.symlink_to(Path(data_dir).resolve(), target_is_directory=True)
        linked = True
    env = dict(os.environ, PYTHONPATH=str(out))
    try:
        for _ in range(rounds):
            files = sorted(
                str(p.relative_to(out)) for p in remaining if p.name.startswith("test_") and p.suffix == ".py"
            )
            if not files:
                break
            result = subprocess.run(
                [
                    python or sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-rfE",
                    "-p",
                    "no:cacheprovider",
                    "-o",
                    "addopts=--import-mode=importlib",
                    "--continue-on-collection-errors",
                    *files,
                ],
                cwd=out,
                env=env,
                capture_output=True,
                text=True,
                timeout=7200,
            )
            failing: dict = {}
            for line in result.stdout.splitlines():
                m = _RESULT_RE.match(line.strip())
                if m and m.group(1) in files:
                    failing.setdefault(m.group(1), (m.group(2) or "error").strip()[:160])
            if not failing:
                if result.returncode not in (0, 5):
                    raise SystemExit(f"release tests did not run cleanly:\n{result.stdout[-3000:]}")
                break
            excluded.update(failing)
            remaining = [p for p in remaining if str(p.relative_to(out)) not in failing]
        else:
            raise SystemExit(f"release tests still fail after {rounds} rounds")
    finally:
        if linked:
            link.unlink()
        for name in {p.name for p in out.iterdir()} - before:
            target = out / name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        for cache in out.rglob("__pycache__"):  # the run byte-compiled the whole tree
            shutil.rmtree(cache, ignore_errors=True)
    return remaining, [f"{path}: {reason}" for path, reason in sorted(excluded.items())]


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
    for problem in unresolved_names(out, staged_modules):
        if problem.startswith("module-level"):
            errors.append(problem)
    for staged_manifest in (out / "logic").rglob("_manifest.py"):
        for key, target in parse_registry_manifest(staged_manifest).items():
            if trimmed(target.split(":")[0], out) not in staged_modules:
                errors.append(f"{staged_manifest.name}: key {key!r} target {target} escapes the tree")
    return errors


def _in_type_checking(tree: ast.Module, target: ast.AST) -> bool:
    """Whether ``target`` sits in an ``if TYPE_CHECKING:`` block."""
    for node in ast.walk(tree):
        guarded = isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"
        if guarded and any(sub is target for stmt in node.body for sub in ast.walk(stmt)):
            return True
    return False


def unresolved_names(out: Path, staged_modules: Set[str]) -> List[str]:
    """``from <staged pkg> import <name>`` statements whose name no longer exists in the staged tree.

    Module-level ones fail on import; deferred ones (in a function or an ImportError guard) fail
    only when that code path runs, which is expected for an excluded feature but worth reading.
    """
    problems: List[str] = []
    for path in sorted((out / "logic").rglob("*.py")):
        if "test" in path.relative_to(out).parts[:2]:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or _in_type_checking(tree, node):
                continue
            targets = _import_targets(node, path, out) or []
            base = targets[-1] if targets else ""
            if not base.startswith("logic") or base not in staged_modules:
                continue
            for target in targets[:-1]:
                if not _import_resolves(target, staged_modules, out):
                    kind = "deferred" if _is_deferred(tree, node) else "module-level"
                    problems.append(f"{kind} import of the pruned name {target} in {path.relative_to(out)}")
    return problems


def unguarded_missing_imports(out: Path, staged_modules: Set[str]) -> List[str]:
    """Imports of ``logic`` modules missing from the staged tree that nothing guards.

    Module-level ones fail verification elsewhere; these are the function-level ones not inside
    a ``try`` that catches ImportError (nor under ``TYPE_CHECKING``): code paths that raise
    ModuleNotFoundError when they run. Each one should be unreachable in the package.
    """
    problems: List[str] = []
    for path in sorted((out / "logic").rglob("*.py")):
        if "test" in path.relative_to(out).parts[:2]:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                targets = [(_import_targets(node, path, out) or [""])[-1]]
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                continue
            for target in targets:
                if not target.startswith("logic") or target in staged_modules:
                    continue
                if _in_type_checking(tree, node) or _guarded_by_import_error(tree, node):
                    continue
                problems.append(
                    f"unguarded import of the pruned module {target} in {path.relative_to(out)}:{node.lineno}"
                )
    return problems


def _guarded_by_import_error(tree: ast.Module, target: ast.AST) -> bool:
    """Whether ``target`` sits in the body of a ``try`` that catches ImportError."""
    for node in ast.walk(tree):
        guards = isinstance(node, ast.Try) and any(_guards_import(handler) for handler in node.handlers)
        if guards and any(sub is target for stmt in node.body for sub in ast.walk(stmt)):
            return True
    return False
