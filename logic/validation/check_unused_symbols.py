"""Triage vulture's unused functions, classes, methods and properties by evidence of use.

vulture finds definitions whose name is never referenced in the scanned Python. In this
codebase many such names are still reached: through registry and Hydra strings
(``module:Class``, ``_target_``), registration decorators, framework callbacks (torch,
Lightning, argparse, ``ast.NodeVisitor``) and overrides of external base classes. This
tool runs vulture over the repository (tests included, so test-only use counts as use),
keeps the findings in production code, and sorts each one into

- ``STRING``: the name is a token in a YAML, JSON, TOML, TypeScript, Rust or notebook
  file, or in a Python string literal (registry targets, Hydra configs, ``getattr`` names);
- ``DECORATED``: the definition has a decorator other than ``staticmethod``,
  ``classmethod``, ``property``, ``abstractmethod`` or ``override`` (registrations);
- ``HOOK``: a callback name a framework calls (``forward``, ``on_*``, ``visit_*``, ...);
- ``OVERRIDE``: a method of a class with an ancestor defined outside the scanned code
  (the external base may call it);
- ``FSTRING``: the name starts or ends with the constant part of an f-string
  (``getattr(self, f"_try_{move}")`` dispatch);
- ``CANDIDATE``: none of the above; no evidence of use anywhere.

Only ``CANDIDATE`` rows are proposed for removal, and each one still needs a look (a
name built at run time escapes every static check). The full test suite is the arbiter.

Example:
    python logic/validation/check_unused_symbols.py --repo .
    python logic/validation/check_unused_symbols.py --repo . --only CANDIDATE --min-confidence 60
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

PRODUCTION = ("logic/src", "logic/controllers", "logic/gen", "logic/package", "logic/validation", "logic/benchmark")
SCAN = ("logic", "main.py", "__main__.py")
EXCLUDE = "*/submodules/*,*/legacy/*,*/archive/*,*/__pycache__/*"
TEXT_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".ts", ".tsx", ".rs", ".ipynb"}
TEXT_ROOTS = ("logic", "app/src", "app/src-tauri/src", "assets/configs", "notebooks")
PLAIN_DECORATORS = {"staticmethod", "classmethod", "property", "abstractmethod", "override", "cached_property"}
HOOK_NAMES = {
    "forward",
    "extra_repr",
    "reset_parameters",
    "training_step",
    "validation_step",
    "test_step",
    "predict_step",
    "configure_optimizers",
    "setup",
    "teardown",
    "prepare_data",
    "train_dataloader",
    "val_dataloader",
    "test_dataloader",
    "state_dict",
    "load_state_dict",
    "generic_visit",
    "format_help",
    "error",
    "emit",
    "filter",
    "run",
    "handle",
    "default",
    "close",
    "flush",
    "write",
    "isatty",
    "fileno",
    "readable",
    "writable",
    "seekable",
    "paintEvent",
    "resizeEvent",
    # torch.nn.Module and TorchRL EnvBase hooks
    "get_extra_state",
    "set_extra_state",
    "_make_spec",
    "_set_seed",
    "_reset",
    "_step",
    "_apply",
    "_load_from_state_dict",
    "_save_to_state_dict",
}
HOOK_PREFIXES = ("on_", "visit_", "test_", "_step", "pytest_", "__")


@dataclass
class Finding:
    kind: str
    name: str
    path: str
    line: int
    last_line: int
    confidence: int
    category: str = "CANDIDATE"


def _text_tokens(repo: Path) -> Set[str]:
    tokens: Set[str] = set()
    for root in TEXT_ROOTS:
        base = repo / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in TEXT_SUFFIXES and "node_modules" not in path.parts:
                try:
                    tokens.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", path.read_text(encoding="utf-8")))
                except (UnicodeDecodeError, OSError):
                    continue
    return tokens


def _python_strings(repo: Path) -> Set[str]:
    """Identifier tokens inside Python string literals (not docstrings) of the scanned code."""
    tokens: Set[str] = set()
    for root in SCAN:
        base = repo / root
        paths: Iterable[Path] = [base] if base.is_file() else base.rglob("*.py")
        for path in paths:
            if any(p in path.parts for p in ("submodules", "__pycache__", "legacy")):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                    tokens.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", node.value))
    return tokens


class _Defs(ast.NodeVisitor):
    """Decorators and class bases by (line, name)."""

    def __init__(self, local_classes: Dict[str, Set[str]]) -> None:
        self.local_classes = local_classes
        self.decorated: Dict[int, List[str]] = {}
        self.external_method_lines: Set[int] = set()

    @staticmethod
    def _decorator_name(node: ast.expr) -> str:
        target = node.func if isinstance(node, ast.Call) else node
        if isinstance(target, ast.Attribute):
            return target.attr
        return getattr(target, "id", "")

    def _record(self, node) -> None:
        names = [self._decorator_name(d) for d in node.decorator_list]
        self.decorated[node.lineno] = names
        for d in node.decorator_list:  # vulture reports the line of the def; decorators sit above it
            self.decorated.setdefault(d.lineno, names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node)
        bases = {b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", "") for b in node.bases}
        if _has_external_ancestor(bases, self.local_classes):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.external_method_lines.add(item.lineno)
        self.generic_visit(node)


def _local_classes(repo: Path) -> Dict[str, Set[str]]:
    """Class name -> base names, for every class of the production code."""
    classes: Dict[str, Set[str]] = {}
    for root in PRODUCTION:
        for path in (repo / root).rglob("*.py") if (repo / root).is_dir() else []:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.ClassDef):
                    bases = {b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", "") for b in n.bases}
                    classes.setdefault(n.name, set()).update(bases)
    return classes


def _has_external_ancestor(bases: Set[str], classes: Dict[str, Set[str]], seen: Optional[Set[str]] = None) -> bool:
    """Some ancestor (by name) is defined outside the scanned code."""
    seen = seen or set()
    for base in bases - {"object", "ABC", "Generic", "Protocol", ""}:
        if base in seen:
            continue
        seen.add(base)
        if base not in classes or _has_external_ancestor(classes[base], classes, seen):
            return True
    return False


def _fstring_fragments(repo: Path) -> Set[str]:
    """Constant prefixes and suffixes of f-strings (``_try_`` of ``f"_try_{name}"``)."""
    fragments: Set[str] = set()
    for root in SCAN:
        base = repo / root
        for path in [base] if base.is_file() else base.rglob("*.py"):
            if any(p in path.parts for p in ("submodules", "__pycache__", "legacy")):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.JoinedStr):
                    for value in node.values:
                        if isinstance(value, ast.Constant) and isinstance(value.value, str):
                            for piece in re.findall(r"[A-Za-z0-9_]+", value.value):
                                if len(piece) >= 3 and ("_" in piece):
                                    fragments.add(piece)
    return fragments


def run_vulture(repo: Path, min_confidence: int) -> List[Finding]:
    from vulture import Vulture

    v = Vulture(verbose=False)
    v.scavenge([str(repo / p) for p in SCAN if (repo / p).exists()], exclude=EXCLUDE.split(","))
    findings = []
    for item in v.get_unused_code(min_confidence=min_confidence):
        if item.typ not in ("function", "class", "method", "property"):
            continue
        rel = Path(item.filename).resolve().relative_to(repo).as_posix()
        if not rel.startswith(PRODUCTION):
            continue
        findings.append(Finding(item.typ, item.name, rel, item.first_lineno, item.last_lineno, item.confidence))
    return findings


def triage(repo: Path, findings: List[Finding]) -> List[Finding]:
    tokens = _text_tokens(repo) | _python_strings(repo)
    fragments = _fstring_fragments(repo)
    local = _local_classes(repo)
    defs: Dict[str, _Defs] = {}
    for f in findings:
        if f.path not in defs:
            visitor = _Defs(local)
            visitor.visit(ast.parse((repo / f.path).read_text(encoding="utf-8")))
            defs[f.path] = visitor
        info = defs[f.path]
        decorators = [d for d in info.decorated.get(f.line, []) if d not in PLAIN_DECORATORS]
        if f.name.startswith("__") and f.name.endswith("__"):
            f.category = "HOOK"  # protocol methods are called by the language
        elif f.name in tokens:
            f.category = "STRING"
        elif decorators:
            f.category = "DECORATED"
        elif f.kind in ("method", "property") and (f.name in HOOK_NAMES or f.name.startswith(HOOK_PREFIXES)):
            f.category = "HOOK"
        elif f.kind in ("method", "property") and f.line in info.external_method_lines:
            f.category = "OVERRIDE"
        elif any(
            (frag.endswith("_") and f.name.startswith(frag)) or (frag.startswith("_") and f.name.endswith(frag))
            for frag in fragments
        ):
            f.category = "FSTRING"
    return findings


def remove(repo: Path, findings: List[Finding]) -> List[str]:
    """Delete ``CANDIDATE`` definitions (with their decorators); returns the files changed."""
    by_file: Dict[str, List[Finding]] = {}
    for f in findings:
        if f.category == "CANDIDATE":
            by_file.setdefault(f.path, []).append(f)
    touched = []
    for rel, items in sorted(by_file.items()):
        path = repo / rel
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        wanted = {(f.name, f.line) for f in items}
        spans = []
        for parent in ast.walk(tree):
            for block_name in ("body", "orelse"):
                block = getattr(parent, block_name, None)
                if not isinstance(block, list):
                    continue
                for node in block:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (node.name, node.lineno) in wanted:
                        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                        others = [n for n in block if n is not node]
                        spans.append((start, node.end_lineno or node.lineno, not others, node.col_offset))
        removed_names = {f.name for f in items}
        # a definition inside another removed one goes with it; deleting both would shift the outer span
        spans = [s for s in spans if not any(o[0] < s[0] and s[1] <= o[1] for o in spans if o is not s)]
        for start, end, empty, col in sorted(spans, reverse=True):
            replacement = [" " * col + "pass\n"] if empty else []
            # also drop the blank lines that separated the definition from the next one
            while end < len(lines) and not lines[end].strip() and not empty:
                end += 1
            lines[start - 1 : end] = replacement
        text = "".join(lines)
        doc = ast.get_docstring(ast.parse(text), clean=False)
        if doc:
            kept = [l for l in doc.splitlines() if not any(re.match(rf"\s+{re.escape(n)}\b.*:", l) for n in removed_names)]
            if len(kept) != len(doc.splitlines()):
                text = text.replace(doc, "\n".join(kept), 1)
        if text != source:
            path.write_text(text, encoding="utf-8")
            touched.append(rel)
    return touched


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--min-confidence", type=int, default=60)
    parser.add_argument("--only", nargs="*", default=None, help="Categories to print (default: all).")
    parser.add_argument("--remove", action="store_true", help="Delete the CANDIDATE definitions (then run ruff).")
    parser.add_argument("--exclude-path", nargs="*", default=[], help="Path prefixes never touched by --remove.")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    findings = triage(repo, run_vulture(repo, args.min_confidence))
    for f in sorted(findings, key=lambda f: (f.category, f.path, f.line)):
        if args.only and f.category not in args.only:
            continue
        print(f"{f.category:9s} {f.kind:8s} {f.path}:{f.line}-{f.last_line} {f.name} ({f.confidence}%)")
    counts = Counter(f.category for f in findings)
    print("; ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if args.remove:
        targets = [f for f in findings if not any(f.path.startswith(p) for p in args.exclude_path)]
        touched = remove(repo, targets)
        print(f"removed definitions from {len(touched)} file(s)")
        for rel in touched:
            print(f"  {rel}")
        return 0
    return 1 if counts.get("CANDIDATE") else 0


if __name__ == "__main__":
    sys.exit(main())
