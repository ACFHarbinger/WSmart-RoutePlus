"""Check that a refactor only moved code: every top-level definition keeps its exact source.

Usage::

    python tools/check_pure_move.py BASE_REV [HEAD_REV] [--paths logic/src ...]

For every Python file under ``--paths`` at ``BASE_REV``, each top-level function,
class and assignment is compared, by name and by source text, with the
definitions found in the same paths at ``HEAD_REV`` (default: the working tree).
A definition that disappears or whose source changes is reported. Import
statements are ignored wherever they appear, including deferred imports inside
function bodies, because moving code rewrites them by design; so are module
docstrings. New definitions in HEAD are listed only with ``--verbose``.

A definition may move to any file, but keeps its name. Names defined in more than
one module are compared module by module, falling back to a match anywhere.

Exit status 0 means every definition of BASE survived unchanged.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

Definitions = Dict[str, Dict[str, str]]  # module path -> name -> normalized source


def _git_files(rev: str, paths: List[str]) -> Dict[str, str]:
    names = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", rev, "--", *paths], capture_output=True, text=True, check=True
    ).stdout.split()
    files = {}
    for name in names:
        if name.endswith(".py"):
            files[name] = subprocess.run(
                ["git", "show", f"{rev}:{name}"], capture_output=True, text=True, check=True
            ).stdout
    return files


def _worktree_files(paths: List[str]) -> Dict[str, str]:
    files = {}
    for root in paths:
        for path in Path(root).rglob("*.py"):
            if "__pycache__" not in path.parts:
                files[str(path)] = path.read_text()
    return files


class _DropImports(ast.NodeTransformer):
    """Remove import statements (module level and deferred) before comparing."""

    def visit_Import(self, node: ast.Import) -> None:
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        return None

    def generic_visit(self, node: ast.AST) -> ast.AST:
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody", "handlers"):
            block = getattr(node, field, None)
            if isinstance(block, list) and field != "handlers" and not block and isinstance(node, ast.stmt):
                block.append(ast.Pass())  # a body that held only imports stays valid
        return node


def _normalized(node: ast.stmt) -> str:
    """The definition's code without import statements, as ``ast.unparse`` prints it."""
    import copy

    clean = _DropImports().visit(copy.deepcopy(node))
    return ast.unparse(ast.fix_missing_locations(clean))


def _names(node: ast.stmt) -> List[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def definitions(files: Dict[str, str]) -> Definitions:
    """Top-level definitions per module, with their decorated source."""
    out: Definitions = {}
    for path, text in files.items():
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise SystemExit(f"{path}: {exc}") from None
        defs = {}
        for node in tree.body:
            names = _names(node)
            if not names or (names == ["__all__"]):
                continue
            if isinstance(node, ast.ClassDef):
                # A class may be split into mixins in other modules: compare each method by
                # name, and the rest of the class body (attributes, nested classes) as one unit.
                rest = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defs[f"{node.name}.{item.name}"] = _normalized(item)
                        defs[f"<method>.{item.name}#{_normalized(item)}"] = _normalized(item)
                    elif not (isinstance(item, ast.Expr) and isinstance(getattr(item, "value", None), ast.Constant)):
                        rest.append(ast.unparse(item))
                defs[f"{node.name}.<body>"] = "\n".join(rest)
                continue
            source = _normalized(node)
            for name in names:
                defs[name] = source
        out[path] = defs
    return out


def compare(base: Definitions, head: Definitions) -> Tuple[List[str], List[str]]:
    """Problems (missing or changed definitions) and new names in HEAD."""
    anywhere: Dict[str, Set[str]] = defaultdict(set)
    for defs in head.values():
        for name, source in defs.items():
            anywhere[name].add(source)
    problems = []
    for path, defs in sorted(base.items()):
        for name, source in defs.items():
            if name.startswith("<method>."):
                continue
            same_module = head.get(path, {}).get(name)
            if same_module == source or source in anywhere.get(name, set()):
                continue
            if "." in name and not name.endswith(".<body>"):
                method = name.split(".", 1)[1]
                if f"<method>.{method}#{source}" in anywhere:
                    continue  # moved to another class (a mixin) unchanged
            if name.endswith(".<body>") and not source:
                continue
            if name in anywhere:
                problems.append(f"CHANGED  {path}::{name}")
            else:
                problems.append(f"MISSING  {path}::{name}")
    base_names = {name for defs in base.values() for name in defs}
    new = sorted(
        f"{path}::{name}"
        for path, defs in head.items()
        for name in defs
        if name not in base_names and not name.startswith("<method>.")
    )
    return problems, new


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("head", nargs="?", default=None, help="Revision to compare (default: the working tree).")
    parser.add_argument("--paths", nargs="+", default=["logic/src"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    base = definitions(_git_files(args.base, args.paths))
    head = definitions(_git_files(args.head, args.paths) if args.head else _worktree_files(args.paths))
    problems, new = compare(base, head)
    for line in problems:
        print(line)
    if args.verbose:
        for line in new:
            print(f"NEW      {line}")
    total = sum(1 for d in base.values() for n in d if not n.startswith("<method>."))
    print(f"{total - len(problems)}/{total} definitions unchanged; {len(problems)} problem(s); {len(new)} new name(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
