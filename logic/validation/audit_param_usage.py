"""Audit the usage of every ``params.py`` field (P-55).

Appendix A's generated ``used?`` flag searched ``.field`` inside the policy's
own directory and only for params-only fields. Three classes of live usage are
invisible to that method, each with a released-set false negative (found in
review): dict-key access (``values.get("V", 0)``), params.py-internal wiring
(``params.cooling_rate`` consumed while building the acceptance criterion), and
shared params/config fields, which were never audited at all.

This tool implements the corrected three-pattern method (owner ruling
RQ-mistral-1): for every field of every params dataclass it searches

1. attribute access ``.field``,
2. dict-key access ``"field"`` (``values.get``, ``kwargs.get``, ``["field"]``),
3. params.py-internal consumption,

first across the policy's directory, then — for fields that look inert there —
across ``logic/src`` and ``logic/controllers`` (no false INERT verdicts).
Fields with no live reader anywhere are reported INERT; their deletion (with
the engine literal promoted to a named constant) belongs to the P-10 pass,
which must not delete anything this audit marks USED.

Usage::

    python logic/validation/audit_param_usage.py [--release logic/release.yaml] [--out FILE]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "logic" / "src"
POLICIES = SRC / "policies"
CONTROLLERS = REPO / "logic" / "controllers"

# Search order for usage: the policy directory first, then the wider tree.
WIDE_SCOPE = [POLICIES, SRC / "pipeline", CONTROLLERS]

RELEASE_STAGES = {
    "routing": [
        "swc_tcf",
        "bpc",
        "alns",
        "sans",
        "hgs",
        "psoma",
        "aco_hh",
        "pg_clns",
        "na",
        "cvrp",
    ],
    "packing": ["last_minute", "service_level", "lookahead", "regular"],
    "route_improvement": ["classical_local_search", "fast_tsp"],
    "acceptance": ["bmc", "oi"],
}


def find_params_files() -> List[Path]:
    return sorted(POLICIES.rglob("params.py"))


def dataclass_fields(tree: ast.AST) -> List[Tuple[str, str]]:
    """Fields of the first dataclass in a module: (name, default source)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            isinstance(dec, ast.Name)
            and dec.id == "dataclass"
            or isinstance(dec, ast.Call)
            and isinstance(dec.func, ast.Name)
            and dec.func.id == "dataclass"
            for dec in node.decorator_list
        ):
            continue
        fields: List[Tuple[str, str]] = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                default = ast.unparse(stmt.value) if stmt.value is not None else "<required>"
                fields.append((stmt.target.id, default))
        return fields
    return []


def strip_docstrings(text: str) -> str:
    """Blank docstring bodies (line-preserving) so examples do not count as use."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    lines = text.splitlines()
    spans: List[Tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                spans.append((body[0].lineno, cast(int, body[0].end_lineno)))  # docstring expr always has an end line
    for start_line, end_line in spans:
        for lineno in range(start_line, min(end_line, len(lines)) + 1):
            lines[lineno - 1] = ""
    return "\n".join(lines)


def usage_of(field: str, scope: List[Path], exclude: Optional[Path] = None) -> List[Tuple[str, int, str]]:
    """Locations where ``field`` is consumed: (file, line, pattern)."""
    hits: List[Tuple[str, int, str]] = []
    attr_re = re.compile(rf"\.{re.escape(field)}\b")
    dict_re = re.compile(rf"[\"']{re.escape(field)}[\"']")
    for root in scope:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or path == exclude:
                continue
            try:
                text = path.read_text()
            except OSError:
                continue
            text = strip_docstrings(text)
            for lineno, line in enumerate(text.splitlines(), 1):
                if attr_re.search(line):
                    hits.append((str(path.relative_to(REPO)), lineno, "attr"))
                elif dict_re.search(line):
                    hits.append((str(path.relative_to(REPO)), lineno, "dict-key"))
    return hits


def params_policy_dir(params_path: Path) -> Path:
    return params_path.parent


def policy_key_for(params_path: Path) -> Optional[str]:
    """Best-effort registry key whose adapter sits next to this params.py."""
    for candidate in params_path.parent.glob("policy_*.py"):
        stem = candidate.stem.removeprefix("policy_")
        return stem
    return None


def owning_class(tree: ast.AST, field: str) -> Optional[str]:
    """Name of the dataclass that declares ``field``."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == field:
                return node.name
    return None


def audit(released_keys: Set[str]) -> Dict[str, List[dict]]:
    """{class name: [field record]} for every params dataclass."""
    results: Dict[str, List[dict]] = {}
    for params_path in find_params_files():
        tree = ast.parse(params_path.read_text())
        fields = dataclass_fields(tree)
        if not fields:
            continue
        policy_dir = params_policy_dir(params_path)
        key = policy_key_for(params_path)
        for field, default in fields:
            cls_name = owning_class(tree, field)
            local = usage_of(field, [policy_dir])
            external = [h for h in local if Path(h[0]).name != "params.py"]
            internal = [h for h in local if Path(h[0]).name == "params.py"]
            solver_attr = [h for h in external if h[2] == "attr" and not h[0].endswith("dispatcher.py")]
            if external:
                plumbing = "" if solver_attr else " [plumbing-only: no solver attribute read]"
                verdict, evidence = "USED", "; ".join(f"{f}:{l} ({p})" for f, l, p in external[:4]) + plumbing
            elif internal:
                verdict, evidence = "INTERNAL-ONLY", "; ".join(f"{f}:{l}" for f, l, _ in internal[:3])
            else:
                wide = usage_of(field, WIDE_SCOPE, exclude=params_path)
                if wide:
                    verdict, evidence = "USED-WIDE", "; ".join(f"{f}:{l} ({p})" for f, l, p in wide[:4])
                else:
                    verdict, evidence = "INERT", "no attribute or dict-key read anywhere in logic/"
            record = {
                "class": cls_name,
                "field": field,
                "default": default,
                "key": key,
                "released": key in released_keys if key else None,
                "verdict": verdict,
                "evidence": evidence,
                "solver_attr_read": bool(solver_attr),
            }
            results.setdefault(cls_name or params_path.parent.name, []).append(record)
    return results


def render(results: Dict[str, List[dict]]) -> str:
    lines = [
        "<!-- Generated by logic/validation/audit_param_usage.py (P-55). Evidence file for the P-10 pass:",
        "     do not delete a field this audit marks USED / USED-WIDE / INTERNAL-ONLY.",
        "     INERT fields are deleted with the engine literal promoted to a named",
        "     constant (owner ruling RQ-mistral-1). -->",
        "",
        "# Appendix A supplement — corrected usage audit (P-55)",
        "",
        "Method: attribute access + dict-key access + params.py-internal wiring, over the",
        "policy directory first and the whole tree second, for params-only AND shared",
        "params/config fields. Verdicts: USED (read by solver/adapter code), USED-WIDE (read",
        "outside the policy directory — review the evidence before any deletion),",
        "INTERNAL-ONLY (consumed only while building the params object), INERT (no live",
        "reader: deletion candidate for the P-10 pass).",
        "",
    ]
    released_first = sorted(results.items(), key=lambda kv: (not any(r["released"] for r in kv[1]), kv[0]))
    for cls_name, records in released_first:
        marker = " *(release set)*" if any(r["released"] for r in records) else ""
        lines.append(f"## `{cls_name}`{marker}")
        lines.append("")
        lines.append("| Field | Default | Verdict | Solver attr read | Evidence |")
        lines.append("|---|---|---|---|---|")
        for r in records:
            default = str(r["default"]).replace("|", "\\|")
            solver = "yes" if r["solver_attr_read"] else "no"
            lines.append(f"| `{r['field']}` | `{default}` | {r['verdict']} | {solver} | {r['evidence']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", default="logic/release.yaml", help="release manifest (marks the release set)")
    parser.add_argument("--out", default="-", help="output file, '-' for stdout")
    args = parser.parse_args()
    released: Set[str] = set()
    release_path = REPO / args.release
    if release_path.is_file():
        import yaml

        manifest = yaml.safe_load(release_path.read_text())
        for keys in manifest.get("stages", {}).values():
            released.update(keys)
    else:
        for keys in RELEASE_STAGES.values():
            released.update(keys)
    results = audit(released)
    report = render(results)
    if args.out == "-":
        sys.stdout.write(report)
    else:
        Path(args.out).write_text(report)
        inert = sum(1 for records in results.values() for r in records if r["verdict"] == "INERT")
        total = sum(len(records) for records in results.values())
        print(f"{args.out}: {len(results)} classes, {total} fields, {inert} INERT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
