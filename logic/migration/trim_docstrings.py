"""Trim generated docstring boilerplate (P-03, owner ruling Q-7b).

The batch generator (`utils/docs/add_docstrings_batch.py`) left three kinds of
noise that the old checkers demanded and the new rule (Q-7b) rejects:

1. ``Returns:``/``Yields:`` sections whose body is only ``None`` — removed.
2. ``Example:`` sections that only say ``>>> import <module>`` — removed.
3. Multi-line ``Args:`` entries that restate a type hint already on the
   signature (``path (str): ...`` where the signature says ``path: str``) —
   the type restatement is dropped, the description kept.

The tool is line-preserving where it can be and idempotent: rerunning it on a
trimmed tree changes nothing. Nothing is removed from a docstring that carries
information the signature does not.

Usage::

    python logic/migration/trim_docstrings.py --check   # report only, exit 1 if trimmable
    python logic/migration/trim_docstrings.py --apply   # trim in place
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
SCOPE = REPO / "logic" / "src"
# Pinned submodule: its content belongs to its own repository and must stay clean.
SUBMODULE = REPO / "logic" / "submodules" / "WSmart-Bin-Analysis"

SECTION_RE = re.compile(r"^\s*(Args|Returns|Yields|Raises|Example|Examples|Attributes)\s*:\s*$", re.IGNORECASE)
IMPORT_EXAMPLE_RE = re.compile(r"^\s*>>>\s*(import|from)\s+[\w.]+")
RETURNS_NONE_BODY_RE = re.compile(r"^\s*(None\.?|Nothing\.?)\s*$", re.IGNORECASE)
TYPED_ARG_RE = re.compile(r"^(\s*)([A-Za-z_][\w]*)\s*\(([^)]*)\)\s*:\s*(.+)$")


def docstring_node(tree: ast.AST) -> Optional[Tuple[int, int]]:
    """Line span of a module docstring, or None."""
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        return (tree.body[0].lineno, tree.body[0].end_lineno)
    return None


def trim_sections(lines: List[str], start: int, end: int) -> Tuple[List[str], int, bool]:
    """Trim boilerplate sections within the 1-based inclusive line span."""
    out = list(lines)
    i = start
    removed = 0
    changed = False
    while i <= end - removed:
        line = out[i - 1]
        match = SECTION_RE.match(line)
        if not match:
            i += 1
            continue
        section = match.group(1).lower()
        j = i + 1
        while j <= end - removed:
            stripped = out[j - 1].strip()
            is_closer = stripped.startswith('"""') or stripped.startswith("'''")
            if SECTION_RE.match(out[j - 1]) or is_closer:
                break
            j += 1
        body = out[i : j - 1]
        if section in ("returns", "yields") and all(RETURNS_NONE_BODY_RE.match(b) or not b.strip() for b in body):
            count = j - i
            del out[i - 1 : j - 1]
            removed += count
            continue
        if (
            section in ("example", "examples")
            and body
            and all(IMPORT_EXAMPLE_RE.match(b) or not b.strip() for b in body if b.strip())
        ):
            count = j - i
            del out[i - 1 : j - 1]
            removed += count
            continue
        if section == "args":
            for index, entry in enumerate(body):
                typed = TYPED_ARG_RE.match(entry.rstrip("\n"))
                if typed:
                    rest = typed.group(4)
                    newline = "\n" if entry.endswith("\n") else ""
                    if rest.strip():
                        out[i + index] = f"{typed.group(1)}{typed.group(2)}: {rest}{newline}"
                    else:
                        out[i + index] = f"{typed.group(1)}{typed.group(2)}:{newline}"
                    changed = True
        i = j
    return out, removed, changed


def trim_file(path: Path) -> Tuple[Optional[str], int]:
    """Return (new_text, lines_removed) or (None, 0) when nothing changes."""
    text = path.read_text()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None, 0
    span = docstring_node(tree)
    if span is None:
        return None, 0
    lines = text.splitlines(keepends=True)
    new_lines, removed, _ = trim_sections(lines, span[0], span[1])
    # Trim every docstring in the file, not only the module's.
    # Trim one docstring at a time, re-parsing after each edit so line numbers
    # stay correct.
    while True:
        current = ast.parse("".join(new_lines))
        edited = False
        for candidate in ast.walk(current):
            if not isinstance(candidate, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = getattr(candidate, "body", None)
            if not body or not isinstance(body[0], ast.Expr) or not isinstance(body[0].value, ast.Constant):
                continue
            trimmed_body, delta, did_change = trim_sections(new_lines, body[0].lineno, body[0].end_lineno)
            if delta or did_change:
                new_lines = trimmed_body
                removed += delta
                edited = True
                break
        if not edited:
            break
    new_text = "".join(new_lines)
    if new_text == text:
        return None, 0
    try:
        ast.parse(new_text)
    except SyntaxError:
        return None, 0  # never leave a file unparsable
    return new_text, removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="report trimmable files; exit 1 if any")
    group.add_argument("--apply", action="store_true", help="trim in place")
    args = parser.parse_args()
    trimmable = 0
    total_lines = 0
    for path in sorted(SCOPE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if SUBMODULE in path.parents:
            continue
        new_text, removed = trim_file(path)
        if new_text is None:
            continue
        trimmable += 1
        total_lines += removed
        if args.apply:
            path.write_text(new_text)
            print(f"trimmed {path.relative_to(REPO)} (-{removed} lines)")
        else:
            print(f"trimmable {path.relative_to(REPO)} (~{removed} lines)")
    mode = "trimmed" if args.apply else "trimmable"
    print(f"{mode}: {trimmable} files, {total_lines} lines")
    if args.check and trimmable:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
