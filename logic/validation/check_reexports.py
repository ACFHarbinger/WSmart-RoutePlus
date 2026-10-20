"""Find compatibility re-exports and point their importers at the defining module.

A **re-export** is a name a non-``__init__`` module imports at module level but never
uses itself: it is there so that ``from that.module import Name`` keeps working after
``Name`` moved. (Package ``__init__`` files are a package's public API and are left alone.)

For every re-export the tool reports

- its **canonical** module: the import chain is followed through other re-exports to the
  module that defines the name, a package ``__init__``, or an external package;
- its **importers**: ``from module import Name`` statements (module-level or deferred,
  absolute or relative) in every scanned file;
- **string references** such as ``mock.patch("module.Name")`` or a YAML ``module:Name``.

With ``--fix`` the importers' statements are rewritten to import from the canonical
module, string references are rewritten to the canonical dotted path, and the re-export
is removed from the module's imports and ``__all__``. Run ``ruff check --select I,F
--fix`` and ``ruff format`` afterwards.

A module whose namespace is read dynamically (``inspect.getmembers(m)``, ``vars(m)``,
``dir(m)``, ``m.__dict__``, ``getattr(m, ...)``, ``importlib.import_module(...)``
of it) keeps its re-exports, which are listed as **dynamic**: removing them would
change what the dynamic reader sees. Imports inside ``try`` or ``if`` blocks (optional
dependencies, ``TYPE_CHECKING``) are not module-level imports and are never touched.

Example:
    python logic/validation/check_reexports.py --repo .
    python logic/validation/check_reexports.py --repo . --fix
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

SCAN_ROOTS = ("logic", "main.py", "__main__.py")
SKIP_PARTS = {"__pycache__", "submodules", "legacy", "archive", "node_modules", ".venv", "html", "target"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".md"}
DYNAMIC_CALLS = {"getmembers", "vars", "dir"}


@dataclass
class Module:
    name: str
    path: Path
    tree: ast.Module
    is_package: bool
    defined: Set[str] = field(default_factory=set)
    imports: Dict[str, Tuple[str, str, ast.ImportFrom]] = field(default_factory=dict)  # alias -> (src, original, node)
    used: Set[str] = field(default_factory=set)


@dataclass
class Reexport:
    module: str
    name: str
    source: str  # where the module imports it from
    original: str
    canonical: str = ""
    importers: List[Tuple[str, int]] = field(default_factory=list)
    strings: List[Tuple[str, int]] = field(default_factory=list)
    attributes: List[Tuple[str, int]] = field(default_factory=list)  # ``import module as m; m.Name``


def module_name(repo: Path, path: Path) -> str:
    parts = list(path.relative_to(repo).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve(module: Module, node: ast.ImportFrom) -> str:
    """The absolute module an ``ImportFrom`` reads from."""
    if not node.level:
        return node.module or ""
    package = module.name.split(".") if module.is_package else module.name.split(".")[:-1]
    base = package[: len(package) - (node.level - 1)] if node.level > 1 else package
    return ".".join([*base, *(node.module.split(".") if node.module else [])])


def iter_files(repo: Path, suffixes: Set[str]) -> Iterable[Path]:
    for root in SCAN_ROOTS:
        start = repo / root
        if start.is_file():
            if start.suffix in suffixes:
                yield start
            continue
        for path in start.rglob("*"):
            if path.is_file() and path.suffix in suffixes and not (SKIP_PARTS & set(path.parts)):
                yield path


def _annotation_strings(tree: ast.AST) -> Set[str]:
    """Identifiers inside string annotations (``"Foo"``, ``"Optional[Foo]"``)."""
    found: Set[str] = set()
    annotations: List[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotations += [a.annotation for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs if a.annotation]
            annotations += [a.annotation for a in (node.args.vararg, node.args.kwarg) if a is not None and a.annotation]
            if node.returns is not None:
                annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
    for annotation in annotations:
        for sub in ast.walk(annotation):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                found.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", sub.value))
    return found


def load(repo: Path) -> Dict[str, Module]:
    modules: Dict[str, Module] = {}
    for path in iter_files(repo, {".py"}):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        mod = Module(module_name(repo, path), path, tree, path.name == "__init__.py")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                mod.defined.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                mod.defined.update(t.id for t in targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
                for alias in node.names:
                    if alias.name != "*":
                        mod.imports[alias.asname or alias.name] = (resolve(mod, node), alias.name, node)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                mod.used.add(node.id)
        mod.used |= _annotation_strings(tree)
        modules[mod.name] = mod
    return modules


def module_bindings(mod: Module, modules: Dict[str, Module]) -> Dict[str, str]:
    """Local names bound to scanned modules (``import a.b as m``, ``from a import b``)."""
    bindings: Dict[str, str] = {}
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.ImportFrom):
            base = resolve(mod, node)
            for alias in node.names:
                target = f"{base}.{alias.name}" if base else alias.name
                if target in modules:
                    bindings[alias.asname or alias.name] = target
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules and alias.asname:
                    bindings[alias.asname] = alias.name
    return bindings


def dynamic_modules(modules: Dict[str, Module]) -> Set[str]:
    """Modules whose namespace some code enumerates or reads by name."""
    dynamic: Set[str] = set()
    for mod in modules.values():
        bindings: Dict[str, str] = {}
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.ImportFrom):
                base = resolve(mod, node)
                for alias in node.names:
                    target = f"{base}.{alias.name}" if base else alias.name
                    if target in modules:  # ``from pkg import module``
                        bindings[alias.asname or alias.name] = target
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in modules:
                        bindings[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Call):
                func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                args = [a.id for a in node.args if isinstance(a, ast.Name)]
                if func in DYNAMIC_CALLS | {"getattr", "hasattr"} and args and args[0] in bindings:
                    dynamic.add(bindings[args[0]])
                if func == "import_module" and node.args and isinstance(node.args[0], ast.Constant):
                    if str(node.args[0].value) in modules:
                        dynamic.add(str(node.args[0].value))
            elif isinstance(node, ast.Attribute) and node.attr == "__dict__" and isinstance(node.value, ast.Name):
                if node.value.id in bindings:
                    dynamic.add(bindings[node.value.id])
    return dynamic


def fstring_patterns(modules: Dict[str, Module]) -> List[Tuple[str, str]]:
    """``(package prefix, name suffix)`` of f-strings that build ``package.{x}:{y}Suffix`` targets."""
    patterns = []
    for mod in modules.values():
        for node in ast.walk(mod.tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            parts = [v.value if isinstance(v, ast.Constant) else None for v in node.values]
            if len(parts) < 2 or not isinstance(parts[0], str) or not parts[0].startswith("logic."):
                continue
            prefix = parts[0].rstrip(".:")
            suffix = parts[-1] if isinstance(parts[-1], str) else ""
            patterns.append((prefix, re.sub(r"^[^A-Za-z_]*", "", suffix.split(":")[-1])))
    return patterns


def built_by_fstring(r: "Reexport", patterns: List[Tuple[str, str]]) -> bool:
    return any(r.module.startswith(prefix) and r.name.endswith(suffix) for prefix, suffix in patterns)


def find_reexports(modules: Dict[str, Module]) -> List[Reexport]:
    found = []
    for mod in modules.values():
        if mod.is_package or mod.name.split(".")[-1] == "__main__":
            continue
        for alias, (source, original, _node) in mod.imports.items():
            if alias not in mod.used and alias not in mod.defined:
                found.append(Reexport(mod.name, alias, source, original))
    return found


def canonical(modules: Dict[str, Module], source: str, name: str, seen: Optional[Set[str]] = None) -> str:
    """Follow ``source``'s own re-export of ``name`` until the defining module."""
    seen = seen or set()
    mod = modules.get(source)
    if mod is None or mod.is_package or name in mod.defined or source in seen:
        return source
    entry = mod.imports.get(name)
    if entry is None:
        return source
    seen.add(source)
    return canonical(modules, entry[0], entry[1], seen)


def collect_importers(modules: Dict[str, Module], targets: Dict[Tuple[str, str], Reexport]) -> None:
    for mod in modules.values():
        bindings = module_bindings(mod, modules)
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bindings:
                hit = targets.get((bindings[node.value.id], node.attr))
                if hit is not None:
                    hit.attributes.append((mod.name, node.lineno))
            if isinstance(node, ast.ImportFrom):
                base = resolve(mod, node)
                for alias in node.names:
                    hit = targets.get((base, alias.name))
                    if hit is not None:
                        hit.importers.append((mod.name, node.lineno))


def collect_strings(repo: Path, targets: Dict[Tuple[str, str], Reexport]) -> None:
    pattern = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*[.:][A-Za-z_][A-Za-z0-9_]*")
    for path in iter_files(repo, TEXT_SUFFIXES):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            for token in pattern.findall(line):
                module, _, name = token.replace(":", ".").rpartition(".")
                hit = targets.get((module, name))
                if hit is not None and not line.lstrip().startswith(("from ", "import ")):
                    hit.strings.append((str(path.relative_to(repo)), number))


# --------------------------------------------------------------------------- fixing


def _render(node: ast.ImportFrom, module: str, names: List[ast.alias], indent: str) -> str:
    parts = [f"{a.name} as {a.asname}" if a.asname else a.name for a in names]
    return f"{indent}from {module} import " + (", ".join(parts) if len(parts) < 4 else "(\n" + "".join(f"{indent}    {p},\n" for p in parts) + f"{indent})")


def _rewrite_imports(mod: Module, moves: Dict[Tuple[str, str], str], drop: Set[str]) -> Optional[str]:
    """Source of ``mod`` with ``(base, name) -> canonical`` imports moved and ``drop`` names removed."""
    lines = mod.path.read_text(encoding="utf-8").splitlines(keepends=True)
    edits: List[Tuple[int, int, str]] = []
    for node in ast.walk(mod.tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        base = resolve(mod, node)
        keep: List[ast.alias] = []
        moved: Dict[str, List[ast.alias]] = defaultdict(list)
        changed = False
        for alias in node.names:
            bound = alias.asname or alias.name
            if node in [mod.imports.get(bound, (None, None, None))[2]] and bound in drop:
                changed = True
                continue
            target = moves.get((base, alias.name))
            if target is not None and target != base:
                moved[target].append(alias)
                changed = True
            else:
                keep.append(alias)
        if not changed:
            continue
        indent = re.match(r"\s*", lines[node.lineno - 1]).group(0)
        source_module = ("." * node.level + (node.module or "")) if node.level else (node.module or "")
        out = []
        if keep:
            out.append(_render(node, source_module, keep, indent))
        out += [_render(node, target, names, indent) for target, names in sorted(moved.items())]
        if not out:
            out = [f"{indent}pass"] if _only_statement(mod.tree, node) else []
        edits.append((node.lineno, node.end_lineno or node.lineno, "\n".join(out) + ("\n" if out else "")))
    if not edits and not drop:
        return None
    for start, end, text in sorted(edits, reverse=True):
        lines[start - 1 : end] = [text] if text else []
    source = "".join(lines)
    if drop:
        source = _drop_from_all(source, drop)
    return source


def _only_statement(tree: ast.AST, node: ast.ImportFrom) -> bool:
    for parent in ast.walk(tree):
        for name in ("body", "orelse", "finalbody"):
            block = getattr(parent, name, None)
            if isinstance(block, list) and node in block:
                return len(block) == 1 and parent is not tree
    return False


def _drop_from_all(source: str, drop: Set[str]) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            if not isinstance(node.value, (ast.List, ast.Tuple)):
                return source
            names = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
            kept = [n for n in names if n not in drop]
            if kept == names:
                return source
            lines = source.splitlines(keepends=True)
            body = "__all__ = [" + ", ".join(repr(n) for n in kept) + "]\n" if kept else ""
            lines[node.lineno - 1 : node.end_lineno] = [body] if body else []
            return "".join(lines)
    return source


def apply_fix(repo: Path, modules: Dict[str, Module], reexports: List[Reexport]) -> List[str]:
    moves = {(r.module, r.name): r.canonical for r in reexports}
    drops: Dict[str, Set[str]] = defaultdict(set)
    for r in reexports:
        drops[r.module].add(r.name)
    touched = []
    for mod in modules.values():
        new = _rewrite_imports(mod, moves, drops.get(mod.name, set()))
        if new is not None and new != mod.path.read_text(encoding="utf-8"):
            mod.path.write_text(new, encoding="utf-8")
            touched.append(str(mod.path.relative_to(repo)))
    by_file: Dict[str, List[Reexport]] = defaultdict(list)
    for r in reexports:
        for path, _ in r.strings:
            by_file[path].append(r)
    for path, items in by_file.items():
        full = repo / path
        text = full.read_text(encoding="utf-8")
        new = text
        for r in items:
            new = re.sub(rf"\b{re.escape(r.module)}([.:]){re.escape(r.name)}\b", rf"{r.canonical}\g<1>{r.name}", new)
        if new != text:
            full.write_text(new, encoding="utf-8")
            touched.append(path)
    return sorted(set(touched))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--fix", action="store_true", help="Rewrite importers and strings; remove the re-exports.")
    parser.add_argument("--only", nargs="*", default=None, help="Limit to modules under these dotted prefixes.")
    parser.add_argument(
        "--ignore-dynamic",
        nargs="*",
        default=[],
        help="Modules whose dynamic reader was checked not to need the re-exports (it filters on __module__).",
    )
    parser.add_argument(
        "--no-strings",
        action="store_true",
        help="With --fix, leave re-exports that a string (mock.patch target, YAML) names; review those by hand.",
    )
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    modules = load(repo)
    dynamic = dynamic_modules(modules) - set(args.ignore_dynamic)
    reexports = [r for r in find_reexports(modules) if not args.only or any(r.module.startswith(p) for p in args.only)]
    patterns = fstring_patterns(modules)
    built = [r for r in reexports if r.module not in dynamic and built_by_fstring(r, patterns)]
    kept = [r for r in reexports if r.module in dynamic]
    reexports = [r for r in reexports if r.module not in dynamic and r not in built]
    for r in sorted(built, key=lambda r: (r.module, r.name)):
        print(f"FSTRING  {r.module}.{r.name} (an f-string may build this target; check and fix by hand)")
    for r in reexports + kept:
        r.canonical = canonical(modules, r.source, r.original)
    targets = {(r.module, r.name): r for r in reexports}
    collect_importers(modules, targets)
    collect_strings(repo, targets)
    attr = [r for r in reexports if r.attributes]
    reexports = [r for r in reexports if not r.attributes]
    for r in sorted(attr, key=lambda r: (r.module, r.name)):
        where = ", ".join(f"{m}:{n}" for m, n in r.attributes[:3])
        print(f"ATTR     {r.module}.{r.name} (used as a module attribute at {where}; fix by hand)")
    for r in sorted(reexports, key=lambda r: (r.module, r.name)):
        print(f"REEXPORT {r.module}.{r.name} -> {r.canonical}.{r.original}  importers={len(r.importers)} strings={len(r.strings)}")
    for r in sorted(kept, key=lambda r: (r.module, r.name)):
        print(f"DYNAMIC  {r.module}.{r.name} (the module's namespace is read dynamically; kept)")
    print(f"{len(reexports)} re-export(s) in {len({r.module for r in reexports})} module(s); {len(kept)} kept (dynamic)")
    if args.fix and args.no_strings:
        reexports = [r for r in reexports if not r.strings]
    if args.fix and reexports:
        touched = apply_fix(repo, modules, reexports)
        print(f"rewrote {len(touched)} file(s)")
        for path in touched:
            print(f"  {path}")
    return 1 if reexports and not args.fix else 0


if __name__ == "__main__":
    sys.exit(main())
