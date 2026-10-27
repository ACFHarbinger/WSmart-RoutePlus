"""Scripted migration of the unreleased policies to the config layer (WP-06, Q-8b).

The released policies moved to one source of truth in WP-04/WP-05: solvers read
the typed config dataclass and ``params.py`` is gone. This script extends the
same migration to the unreleased policies, mechanically:

- **Group A/B/D** (typed config wired in the adapter): params-only fields are
  appended to the config dataclass (with their params defaults; default
  conflicts on shared fields are logged, not silently resolved), ``params.py``
  is deleted, and every ``XParams`` reference is rewritten to ``<Key>Config``.
- **Group C** (no config anywhere): the params dataclass moves to
  ``logic/src/configs/policies/<key>.py`` as ``<Key>Config`` (``from_config``
  dropped).
- **Group C-unwired** (a config file exists but the adapter returns ``None``):
  params-only fields merge into the existing config dataclass and the adapter is
  wired (``_config_class()`` returns it; ``XParams.from_config(values)`` becomes
  ``self._effective_config(values)``).

Files the script cannot handle mechanically (multi-class params modules,
multi-config config files, non-matching class names) are skipped and reported —
they are the hand-fix queue for the Q-8b decision point.

Usage::

    python logic/migration/migrate_unreleased_params.py --plan    # report, no changes
    python logic/migration/migrate_unreleased_params.py --apply   # migrate + report
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

REPO = Path(__file__).resolve().parents[2]
LOGIC = REPO / "logic"
POLICIES = LOGIC / "src" / "policies"
CONFIGS = LOGIC / "src" / "configs" / "policies"
CONFIGS_INIT = CONFIGS / "__init__.py"

REWRITE_SCOPE = [LOGIC / "src" / "policies", LOGIC / "src" / "configs", LOGIC / "src" / "pipeline", LOGIC / "test"]


def stage_manifests() -> Dict[str, str]:
    """Registry key -> 'module:Class' across the stage manifests."""
    entries: Dict[str, str] = {}
    for manifest in POLICIES.glob("*/_manifest.py"):
        for key, target in re.findall(r"'([a-z0-9_-]+)':\s*'(logic\.[\w.]+:\w+)'", manifest.read_text()):
            entries[key] = target
    return entries


def is_dataclass(node: ast.ClassDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "dataclass":
            return True
        if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "dataclass":
            return True
    return False


def dataclasses_of(tree: ast.AST) -> List[ast.ClassDef]:
    body = cast(ast.Module, tree).body  # callers pass ast.parse results
    return [n for n in body if isinstance(n, ast.ClassDef) and is_dataclass(n)]


def dataclasses_anywhere(tree: ast.AST) -> List[ast.ClassDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and is_dataclass(n)]


def class_fields(node: ast.ClassDef) -> List[str]:
    return [
        stmt.target.id for stmt in node.body if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]


def config_class_name_for(adapter: Optional[Path]) -> Optional[str]:
    """The config class name an adapter's ``_config_class()`` returns (static).

    Parsing instead of importing keeps the planner independent of the tree's
    import health: a half-migrated module cannot poison the plan.
    """
    if adapter is None:
        return None
    tree = ast.parse(adapter.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_config_class":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
                    return stmt.value.id
                if isinstance(stmt, ast.Return) and stmt.value is None:
                    return None
    return None


def snake_to_pascal(key: str) -> str:
    return "".join(part.title() for part in key.split("_"))


def params_module_import_lines(params_path: Path) -> List[str]:
    """Top-level import statements of a params module (no leading whitespace)."""
    tree = ast.parse(params_path.read_text())
    params_path.read_text().splitlines()
    imports: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.append("import " + ", ".join(a.name for a in node.names))
        elif isinstance(node, ast.ImportFrom):
            module = "." * (node.level or 0) + (node.module or "")
            imports.append(f"from {module} import " + ", ".join(a.name for a in node.names))
    return imports


def field_source_lines(params_path: Path, params_cls: ast.ClassDef, names: List[str]) -> List[str]:
    """Full source of the given fields (multi-line definitions included)."""
    lines = params_path.read_text().splitlines()
    out: List[str] = []
    for stmt in params_cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id in names:
            segment = lines[stmt.lineno - 1 : stmt.end_lineno]
            out.extend(line.rstrip() for line in segment)
    return out


def migration_targets() -> List[dict]:
    """Every remaining params.py with its migration decision."""
    manifests = stage_manifests()
    key_overrides = {"adp": "adp_rollout"}
    targets: List[dict] = []
    for params_path in sorted(POLICIES.rglob("params.py")):
        policy_file = next(params_path.parent.glob("policy_*.py"), None)
        key = policy_file.stem.removeprefix("policy_") if policy_file else params_path.parent.name
        key = key_overrides.get(key, key)
        target = manifests.get(key)
        record = {
            "key": key,
            "params_path": params_path,
            "adapter": policy_file,
            "registry_target": target,
            "group": None,
            "params_class": None,
            "config_cls": None,
            "config_name": None,
            "config_path": None,
            "skip": None,
        }
        if target is None:
            record["skip"] = "no registry key"
            targets.append(record)
            continue
        cfg_name = config_class_name_for(policy_file)
        tree = ast.parse(params_path.read_text())
        classes = dataclasses_anywhere(tree)
        if len(classes) != 1:
            record["skip"] = (
                f"{len(classes)} dataclasses in params.py " + ", ".join(c.name for c in classes) + " (hand-fix)"
            )
            targets.append(record)
            continue
        record["params_class"] = classes[0].name
        record["tree"] = tree
        existing_config = CONFIGS / f"{key}.py"
        if cfg_name is not None and cfg_name != classes[0].name:
            # Wired config: merge params-only fields into it.
            record["group"] = "A/B/D"
            config_path = None
            for candidate in CONFIGS.glob("*.py"):
                if re.search(rf"class {cfg_name}\b", candidate.read_text()):
                    config_path = candidate
                    break
            if config_path is None:
                record["skip"] = f"config {cfg_name} source file not found"
            else:
                record["config_path"] = config_path
                record["config_name"] = cfg_name
        elif cfg_name is not None and cfg_name == classes[0].name:
            # The params class doubles as the config (e.g. na).
            record["group"] = "C-unwired" if existing_config.is_file() else "C"
        else:
            # Adapter returns None.
            record["group"] = "C-unwired" if existing_config.is_file() else "C"
        if record["group"] == "C-unwired":
            config_tree = ast.parse(existing_config.read_text())
            named = [c for c in dataclasses_anywhere(config_tree) if "Config" in c.name]
            if len(named) != 1:
                record["skip"] = f"configs/policies/{key}.py has {len(named)} config dataclasses (hand-fix)"
            else:
                record["config_name"] = named[0].name
                record["config_path"] = existing_config
        targets.append(record)
    return targets


def merge_fields_into_config(record: dict, report: List[str]) -> bool:
    """Append params-only fields to the existing config dataclass."""
    params_path: Path = record["params_path"]
    params_tree: ast.AST = record["tree"]
    params_cls = next(c for c in dataclasses_anywhere(params_tree) if c.name == record["params_class"])
    config_path: Path = record["config_path"]
    config_cls_name = record["config_name"]
    config_text = config_path.read_text()
    config_tree = ast.parse(config_text)
    config_node = next(c for c in dataclasses_anywhere(config_tree) if c.name == config_cls_name)
    config_field_names = set(class_fields(config_node))
    params_field_names = set(class_fields(params_cls))
    extra = params_field_names - config_field_names
    shared = params_field_names & config_field_names
    if shared:
        report.append(f"{record['key']}: {len(shared)} shared fields with the config (defaults kept as-is)")
    if not extra:
        report.append(f"{record['key']}: params is a subset of the config; no field merge")
        return True
    new_fields = field_source_lines(params_path, params_cls, sorted(extra))
    # Insert before __post_init__, else after the last class-body line.
    insert_line = None
    for stmt in config_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__post_init__":
            insert_line = stmt.lineno
            break
    if insert_line is None:
        insert_line = cast(int, config_node.end_lineno) + 1  # parsed class always has an end line
    out = config_text.splitlines()
    out[insert_line - 1 : insert_line - 1] = new_fields
    # Carry the params module's top-level imports for the appended field types:
    # insert the block once, right after the docstring / __future__ imports.
    existing = set(out)
    has_future = any("__future__" in line for line in out)
    imports = [imp for imp in params_module_import_lines(params_path) if imp not in existing]
    future = [] if has_future else [imp for imp in imports if imp.startswith("from __future__")]
    regular = [imp for imp in imports if not imp.startswith("from __future__")]
    if future or regular:
        tree = ast.parse("\n".join(out))
        docstring_end = (
            cast(int, tree.body[0].end_lineno) if tree.body and isinstance(tree.body[0], ast.Expr) else 0
        )
        last_import_end = docstring_end
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                last_import_end = cast(int, node.end_lineno)  # import statements always have an end line
            elif isinstance(node, ast.Expr) and tree.body and node is tree.body[0]:
                continue
            else:
                break
        if future:
            out[docstring_end:docstring_end] = future
        if regular:
            insert_pos = max(last_import_end, docstring_end) + len(future)
            out[insert_pos:insert_pos] = regular
    config_path.write_text("\n".join(out) + "\n")
    report.append(f"{record['key']}: appended {len(new_fields)} params-only field lines to {config_cls_name}")
    return True


def create_config_from_params(record: dict, report: List[str]) -> bool:
    """Group C: move the params dataclass to configs/policies/<key>.py."""
    key = record["key"]
    params_path: Path = record["params_path"]
    params_cls = next(c for c in dataclasses_anywhere(record["tree"]) if c.name == record["params_class"])
    config_name = f"{snake_to_pascal(key)}Config"
    lines = params_path.read_text().splitlines(keepends=True)
    # Drop the from_config classmethod (and its decorator) by AST span.
    spans: List[Tuple[int, int]] = []
    for node in ast.walk(record["tree"]):
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "from_config":
                    start = stmt.lineno
                    for dec in stmt.decorator_list:
                        start = min(start, dec.lineno)
                    spans.append((start, cast(int, stmt.end_lineno)))  # function statements always have an end line
    # Strip the params module's own docstring and imports: the generated file
    # gets a fresh header, and the imports are re-added cleanly below.
    module_spans: List[Tuple[int, int]] = list(spans)
    body_first = record["tree"].body[0]
    if isinstance(body_first, ast.Expr):
        module_spans.append((body_first.lineno, cast(int, body_first.end_lineno)))
    for node in record["tree"].body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_spans.append((node.lineno, cast(int, node.end_lineno)))
    body_lines = [
        line for lineno, line in enumerate(lines, 1) if not any(start <= lineno <= end for start, end in module_spans)
    ]
    body = "".join(body_lines)
    body = re.sub(rf"class {params_cls.name}\b", f"class {config_name}", body)
    body = body.strip() + "\n"
    # Re-indent to module level if the class was nested.
    if body.startswith(" ") or body.startswith("\t"):
        dedent = len(body) - len(body.lstrip())
        body = "\n".join(line[dedent:] if line.startswith(" " * dedent) else line for line in body.splitlines())
    header = (
        f'"""{config_name}: configuration dataclass for the ``{key}`` policy.\n\n'
        "Moved from the policy's ``params.py`` by the WP-06 scripted migration"
        "(Q-8b): the config dataclass is the single source of truth (P-10).\n"
        '"""\n\nfrom __future__ import annotations\n\n'
    )
    imports = params_module_import_lines(params_path)
    imports_text = ("\n".join(imports) + "\n\n") if imports else ""
    target_path = CONFIGS / f"{key}.py"
    if target_path.exists():
        report.append(f"{key}: config file {target_path.name} already exists; skipped")
        return False
    target_path.write_text(header + imports_text + body)
    report.append(f"{key}: created configs/policies/{key}.py with {config_name}")
    init_text = CONFIGS_INIT.read_text()
    if f"from logic.src.configs.policies.{key} import" not in init_text:
        init_text = init_text.replace(
            "__all__ = [",
            f"from logic.src.configs.policies.{key} import {config_name}\n\n__all__ = [",
            1,
        )
        init_text = init_text.replace("__all__ = [", f'__all__ = [\n    "{config_name}",', 1)
        CONFIGS_INIT.write_text(init_text)
        report.append(f"{key}: registered {config_name} in configs/policies/__init__.py")
    record["config_name"] = config_name
    record["config_path"] = target_path
    return True


def adapter_config_class_update(record: dict, report: List[str]) -> None:
    """Point the adapter's _config_class() at the config."""
    adapter: Optional[Path] = record["adapter"]
    if adapter is None or record["config_name"] is None:
        return
    text = adapter.read_text()
    new_text = re.sub(
        r"(def _config_class\(cls\)[^\n]*\n(?:\s+\"\"\".*?\"\"\"\n)?)(\s+)return None",
        rf"\1\2return {record['config_name']}",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if new_text != text:
        adapter.write_text(new_text)
        report.append(f"{record['key']}: _config_class() returns {record['config_name']}")


def rewrite_references(record: dict, report: List[str]) -> List[str]:
    """Rewrite XParams -> <Key>Config across the tree; return manual-fix sites."""
    key = record["key"]
    params_cls = record["params_class"]
    config_name = record["config_name"]
    if config_name is None:
        return []
    module = f"logic.src.configs.policies.{key}"
    manual: List[str] = []
    pattern = re.compile(rf"\b{params_cls}\b")
    for root in REWRITE_SCOPE:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text()
            if params_cls not in text:
                continue
            new_text = pattern.sub(config_name, text)
            if path == record["adapter"]:
                new_text = re.sub(
                    rf"{config_name}\.from_config\((\w+)\)",
                    r"self._effective_config(\1)",
                    new_text,
                )
            new_text = re.sub(
                rf"from (?:logic\.src\.policies\.[\w.]+|\.)\.?params import {config_name}",
                f"from {module} import {config_name}",
                new_text,
            )

            if new_text != text:
                path.write_text(new_text)
            leftovers = [
                f"{path.relative_to(REPO)}:{lineno}"
                for lineno, line in enumerate(new_text.splitlines(), 1)
                if ".from_config(" in line and config_name in line
            ]
            manual.extend(leftovers)
    return manual


def delete_params(record: dict, report: List[str]) -> None:
    record["params_path"].unlink()
    report.append(f"{record['key']}: deleted {record['params_path'].relative_to(REPO)}")


def run(apply: bool) -> int:
    report: List[str] = []
    manual_fixes: List[str] = []
    targets = migration_targets()
    migrated = 0
    for record in targets:
        if record["skip"]:
            report.append(f"SKIP {record['key']}: {record['skip']}")
            continue
        if not apply:
            report.append(f"PLAN {record['key']}: group {record['group']}, class {record['params_class']}")
            continue
        ok = True
        if record["group"] == "C":
            ok = create_config_from_params(record, report)
        else:
            ok = merge_fields_into_config(record, report)
        if ok:
            adapter_config_class_update(record, report)
            manual_fixes += rewrite_references(record, report)
            delete_params(record, report)
            migrated += 1
    mode = "migrated" if apply else "planned"
    print(f"{mode} {migrated} of {len(targets)} params modules")
    for line in report:
        print(f"  - {line}")
    if manual_fixes:
        print("MANUAL from_config sites left (rewrite by hand):")
        for site in manual_fixes:
            print(f"  - {site}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    return run(apply=args.apply)


if __name__ == "__main__":
    sys.exit(main())
