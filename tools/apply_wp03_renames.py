#!/usr/bin/env python3
"""P-45 leftover renames for WP-03. Run from the worktree root after convert_policy_yaml.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    "rl4co",
}
SKIP_PATH_PARTS = {
    ".agent/tasks",
    ".agent/cache",
    ".agent/reports",
    "assets/model_weights",
    "assets/papers/A-Comprehensive-Combinatorial-Optimization-Algorithms-Analysis",
}

TEXT_SUFFIXES = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".toml",
    ".txt",
    ".tex",
    ".j2",
    ".rst",
}

# Longer tokens first.
REPLACEMENTS = [
    ("list_available_packing_strategies", "list_available_packing_strategies"),
    ("routing_labels", "routing_labels"),
    ("ROUTING_SPEC", "ROUTING_SPEC"),
    ("table_routing", "table_routing"),
    ("results_routing.tex", "results_routing.tex"),
    ("results_routing", "results_routing"),
    ("SearchPhase.PACKING", "SearchPhase.PACKING"),
    ("SearchPhase.ROUTING", "SearchPhase.ROUTING"),
    ("SELECTION = \"selection\"", "PACKING = \"packing\""),
    ("CONSTRUCTION = \"construction\"", "ROUTING = \"routing\""),
    ("PackingContext", "PackingContext"),
    ("packing_context.py", "packing_context.py"),
    ("packing_context", "packing_context"),
    ("factory", "factory"),
    ("registry", "registry"),
    (".pack(", ".pack("),
    ("def pack(", "def pack("),
    ("pack", "pack"),
    ("other/pk_", "other/pk_"),
    ("file_prefix=\"ms_\"", "file_prefix=\"pk_\""),
    ("file_prefix='pk_'", "file_prefix='pk_'"),
    ("configs_dir, \"ms_\"", "configs_dir, \"pk_\""),
    ("configs_dir, 'pk_'", "configs_dir, 'pk_'"),
    ("prefixed ``pk_``", "prefixed ``pk_``"),
    ("pk_yaml", "pk_yaml"),
    ("pk_*.yaml", "pk_*.yaml"),
    ("glob(\"ms_", "glob(\"pk_"),
    ("glob('pk_", "glob('pk_"),
    ("f\"ms_{", "f\"pk_{"),
    ("f'pk_{", "f'pk_{"),
]


def skip_path(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    return any(rel.startswith(prefix) for prefix in SKIP_PATH_PARTS)


def git_mv(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "mv", str(src), str(dest)], cwd=ROOT)


def rename_packing_modules() -> None:
    packing = ROOT / "logic" / "src" / "policies" / "packing"
    for src in sorted(packing.glob("selection_*.py")):
        dest = src.with_name(src.name[len("selection_") :])
        git_mv(src, dest)
    base = packing / "base"
    if (base / "factory.py").exists():
        git_mv(base / "factory.py", base / "factory.py")
    if (base / "registry.py").exists():
        git_mv(base / "registry.py", base / "registry.py")
    ctx = ROOT / "logic" / "src" / "interfaces" / "context" / "packing_context.py"
    if ctx.exists():
        git_mv(ctx, ctx.with_name("packing_context.py"))


def rewrite_module_paths(text: str) -> str:
    text = text.replace(
        "logic.src.policies.packing.",
        "logic.src.policies.packing.",
    )
    text = text.replace(
        "logic.src.policies.packing.base.factory",
        "logic.src.policies.packing.base.factory",
    )
    text = text.replace(
        "logic.src.policies.packing.base.registry",
        "logic.src.policies.packing.base.registry",
    )
    text = text.replace(
        "from logic.src.policies.packing.",
        "from logic.src.policies.packing.",
    )
    text = text.replace(
        "logic.src.interfaces.context.packing_context",
        "logic.src.interfaces.context.packing_context",
    )
    return text


def rewrite_text(text: str, rel: str) -> str:
    # Do not rename Multi-Stage BPC identifiers.
    placeholders = {
        "ms_bpc_sp": "ms_bpc_sp",
        "ms_bpc": "ms_bpc",
        "run_ms_bpc": "run_ms_bpc",
        "MS_BPC": "MS_BPC",
    }
    for src, token in placeholders.items():
        text = text.replace(src, token)
    text = rewrite_module_paths(text)
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    # SRC/ARCO pool field in Python/YAML/JSON — not generic English "constructors".
    if rel.endswith((".py", ".yaml", ".yml", ".json")):
        text = text.replace("params.routing_algorithms", "params.routing_algorithms")
        text = text.replace("config.routing_algorithms", "config.routing_algorithms")
        text = text.replace('key == "routing_algorithms"', 'key == "routing_algorithms"')
        text = text.replace('"routing_algorithms" in', '"routing_algorithms" in')
        text = text.replace("'routing_algorithms' in", "'routing_algorithms' in")
        text = text.replace('merged.get("routing_algorithms")', 'merged.get("routing_algorithms")')
        text = text.replace("self.params.routing_algorithms", "self.params.routing_algorithms")
    for src, token in placeholders.items():
        text = text.replace(token, src)
    return text


def rewrite_tree() -> int:
    changed = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or skip_path(path):
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        rel = path.relative_to(ROOT).as_posix()
        original = path.read_text(encoding="utf-8")
        updated = rewrite_text(original, rel)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def main() -> None:
    rename_packing_modules()
    changed = rewrite_tree()
    print(f"rewrote {changed} files")


if __name__ == "__main__":
    main()
