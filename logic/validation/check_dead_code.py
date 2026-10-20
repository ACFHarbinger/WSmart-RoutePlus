"""Find unused modules and compatibility alias re-exports under ``logic/src``.

A module is live when another module imports it, when its dotted path or a
``module:name`` target appears in Python, YAML, or JSON, or when an f-string
builds that path from a module-level string constant (Hydra registries).
``__main__.py`` is an entry point.

A module-level ``Alias = Canonical`` binding, or ``from module import
Canonical as Alias``, is a shim when nothing imports ``Alias`` from that
module and no string names ``module.Alias`` or ``module:Alias``. Assignments
to ``Any`` or ``object`` are missing-dependency fallbacks, not shims. Bare
names are not enough: the same class name exists in more than one package.

Example:
    python logic/validation/check_dead_code.py logic/src --repo .
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "legacy",
    "submodules",
    "archive",
}

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json"}
_FALLBACK_CANONICAL = {"Any", "object"}
_TRAILING_ATTR = re.compile(r"^[:.][A-Za-z_][A-Za-z0-9_.]*$")


def _module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(package: str, module: Optional[str], level: int) -> str:
    """Resolve ``from .{module}`` against ``__package__``.

    ``level`` counts dots. One dot stays in the current package.
    """
    parts = package.split(".") if package else []
    climb = max(level - 1, 0)
    if climb > len(parts):
        base: List[str] = []
    else:
        base = parts[: len(parts) - climb]
    if module:
        base.extend(module.split("."))
    return ".".join(part for part in base if part)


def _iter_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen: Set[Path] = set()
    for start in roots:
        if not start.exists():
            continue
        for path in start.rglob("*"):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix not in TEXT_SUFFIXES or not path.is_file():
                continue
            if path in seen:
                continue
            seen.add(path)
            yield path


def _string_blob(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _const_text(node: ast.AST, constants: Dict[str, str]) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _fstring_refs(node: ast.JoinedStr, constants: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """Return exact strings and dynamic module prefixes built by one f-string."""
    chunks: List[Optional[str]] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            chunks.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            chunks.append(_const_text(value.value, constants))
        else:
            chunks.append(None)
    if all(chunk is not None for chunk in chunks):
        return ["".join(chunk or "" for chunk in chunks)], []
    if chunks.count(None) != 1:
        return [chunk for chunk in chunks if chunk and chunk.startswith("logic.")], []
    index = chunks.index(None)
    left = "".join(chunk or "" for chunk in chunks[:index])
    right = "".join(chunk or "" for chunk in chunks[index + 1 :])
    prefixes: List[str] = []
    if (
        left.endswith(".")
        and left.startswith("logic.src.")
        and left.count(".") >= 4
        and (right == "" or _TRAILING_ATTR.match(right))
    ):
        prefixes.append(left)
    exact = [chunk for chunk in (left, right) if chunk.startswith("logic.")]
    return exact, prefixes


class _ImportVisitor(ast.NodeVisitor):
    """Collect imports, alias bindings, and f-string targets from one module."""

    def __init__(self, package: str) -> None:
        self.package = package
        self.imports: Set[str] = set()
        self.imported_names: List[Tuple[str, str]] = []
        self.aliases: List[Tuple[str, str, int]] = []
        self.constants: Dict[str, str] = {}
        self.exact_refs: List[str] = []
        self.prefix_refs: List[str] = []
        self.loaded: Set[str] = set()
        self.bound_modules: Dict[str, str] = {}
        self._depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name)
            local = alias.asname or alias.name.split(".")[0]
            self.bound_modules[local] = alias.name if alias.asname else alias.name.split(".")[0]
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = _resolve_relative(self.package, node.module, node.level) if node.level else node.module or ""
        if module:
            self.imports.add(module)
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            if module:
                self.imported_names.append((module, alias.name))
                self.bound_modules[local] = f"{module}.{alias.name}"
            if (
                self._depth == 0
                and alias.asname
                and alias.asname != alias.name
                and alias.name not in _FALLBACK_CANONICAL
            ):
                self.aliases.append((alias.asname, alias.name, node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._depth == 0 and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                self.constants[target] = node.value.value
            elif isinstance(node.value, ast.Name) and node.value.id not in _FALLBACK_CANONICAL:
                self.aliases.append((target, node.value.id, node.lineno))
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            bound = self.bound_modules.get(node.value.id)
            if bound:
                self.imported_names.append((bound, node.attr))
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        exact, prefixes = _fstring_refs(node, self.constants)
        self.exact_refs.extend(exact)
        self.prefix_refs.extend(prefixes)
        self.generic_visit(node)


def _package_of(full: str, path: Path) -> str:
    if path.name == "__init__.py":
        return full
    return full.rsplit(".", 1)[0] if "." in full else ""


def _parse(path: Path, repo: Path) -> Optional[_ImportVisitor]:
    full = _module_name(repo, path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None
    visitor = _ImportVisitor(_package_of(full, path))
    visitor.visit(tree)
    return visitor


def _mark_import(inbound: Dict[str, Set[str]], target: str, source: str) -> None:
    inbound.setdefault(target, set()).add(source)
    parent = target
    while "." in parent:
        parent = parent.rsplit(".", 1)[0]
        inbound.setdefault(parent, set()).add(source)


def _scan_importers(
    modules: Dict[str, Path],
    repo: Path,
    importer_roots: Iterable[Path],
) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]], List[str], List[str], List[Tuple[str, str, str, int]]]:
    inbound: Dict[str, Set[str]] = {name: set() for name in modules}
    imported_from: Set[Tuple[str, str]] = set()
    resolved_text: List[str] = []
    prefixes: List[str] = []
    aliases: List[Tuple[str, str, str, int]] = []
    seen: Set[Path] = set()
    module_paths = set(modules.values())
    for start in importer_roots:
        if not start.exists():
            continue
        candidates = start.rglob("*.py") if start.is_dir() else [start]
        for path in candidates:
            if any(part in SKIP_DIRS for part in path.parts) or path in seen:
                continue
            seen.add(path)
            full = _module_name(repo, path)
            visitor = _parse(path, repo)
            if visitor is None:
                continue
            for target in visitor.imports:
                _mark_import(inbound, target, full)
            imported_from.update(visitor.imported_names)
            resolved_text.extend(visitor.exact_refs)
            prefixes.extend(visitor.prefix_refs)
            if path not in module_paths:
                continue
            for alias, canonical, lineno in visitor.aliases:
                if alias == canonical or alias == "__all__" or alias in visitor.loaded:
                    continue
                aliases.append((full, alias, canonical, lineno))
    return inbound, imported_from, resolved_text, prefixes, aliases


def _unused_modules(
    modules: Dict[str, Path],
    repo: Path,
    inbound: Dict[str, Set[str]],
    blob: str,
    prefixes: List[str],
) -> List[str]:
    unused: List[str] = []
    for module, path in sorted(modules.items()):
        if path.name in {"__init__.py", "__main__.py"}:
            continue
        if path.name.endswith("_copy.py"):
            continue
        callers = {item for item in inbound.get(module, set()) if item != module}
        if callers or module in blob or any(module.startswith(prefix) for prefix in prefixes):
            continue
        unused.append(f"{module} ({path.relative_to(repo)})")
    return unused


def analyse(root: Path, repo: Path) -> Tuple[List[str], List[str]]:
    """Return unused module paths and shim descriptions."""
    py_files = [path for path in root.rglob("*.py") if not any(part in SKIP_DIRS for part in path.parts)]
    modules = {_module_name(repo, path): path for path in py_files}
    importer_roots = [
        root,
        repo / "logic" / "test",
        repo / "logic" / "controllers",
        repo / "logic" / "validation",
        repo / "main.py",
    ]
    inbound, imported_from, resolved_text, prefixes, aliases = _scan_importers(modules, repo, importer_roots)
    blob = "\n".join([*_iter_text(repo), *resolved_text])
    unused = _unused_modules(modules, repo, inbound, blob, prefixes)
    shims = [
        f"{module}:{lineno} {alias} = {canonical}"
        for module, alias, canonical, lineno in aliases
        if (module, alias) not in imported_from
        and f"{module}.{alias}" not in blob
        and f"{module}:{alias}" not in blob
    ]
    return unused, shims


def _iter_text(repo: Path) -> Iterable[str]:
    for path in _iter_files([repo / "logic", repo / "assets"]):
        yield _string_blob(path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Package root to scan, usually logic/src.")
    parser.add_argument("--repo", type=Path, default=None, help="Repository root used to resolve dotted names.")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    repo = (args.repo or root).resolve()
    if not (repo / "logic").is_dir():
        cursor = root
        while cursor.parent != cursor:
            if (cursor / "logic" / "src").is_dir() and cursor.name != "logic":
                repo = cursor
                break
            cursor = cursor.parent
    unused, shims = analyse(root, repo if (repo / "logic").is_dir() else root.parents[1])
    print(f"unused modules: {len(unused)}")
    for line in unused:
        print(f"  UNUSED {line}")
    print(f"alias shims: {len(shims)}")
    for line in shims:
        print(f"  SHIM {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
