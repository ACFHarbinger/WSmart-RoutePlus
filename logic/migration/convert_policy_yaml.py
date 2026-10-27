#!/usr/bin/env python3
"""One-shot P-13 / P-54 converter for policy YAML, packing YAML, and HPO jobs.

Converts list-of-single-key-maps to plain mappings, remaps packing field names
to ``threshold`` / ``combined_strategies``, rewrites ``ms_`` file refs to
``pk_``, and renames SRC/ARCO ``constructors`` to ``routing_algorithms``.

Does not move ``start_temp`` / ``cooling_rate`` into acceptance (P-11).
Does not add HPO key-liveness CI (P-56).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, cast

try:
    from ruamel.yaml import YAML  # pyrefly: ignore[missing-import]  # optional ruamel dependency
    from ruamel.yaml.comments import (  # pyrefly: ignore[missing-import]  # optional ruamel dependency
        CommentedMap,
        CommentedSeq,
    )

    _HAS_RUAMEL = True
except ImportError:  # pragma: no cover
    YAML = None  # type: ignore[misc, assignment]
    CommentedMap = dict  # type: ignore[misc, assignment]
    CommentedSeq = list  # type: ignore[misc, assignment]
    _HAS_RUAMEL = False

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ROOT / "logic" / "configs" / "policies"
OTHER = POLICIES / "other"
HPO_JOBS = ROOT / "logic" / "src" / "pipeline" / "simulations" / "hpo" / "routing"

PACKING_FIELD_REMAP: Dict[str, str] = {
    "frequency": "threshold",
    "confidence_factor": "threshold",
    "learned_threshold": "threshold",
    "n_sectors": "threshold",
    "period": "threshold",
    "strategies": "combined_strategies",
}

SIDE_KEYS = frozenset({"packing", "route_improvement", "acceptance_criteria", "acceptance_criterion", "strategy"})


def _is_single_key_map_list(node: Any) -> bool:
    if not isinstance(node, list) or not node:
        return False
    for item in node:
        if not isinstance(item, Mapping) or len(item) != 1:
            return False
        key = next(iter(item))
        if key == "strategy":
            return False
    return True


def _new_map() -> Any:
    return CommentedMap() if _HAS_RUAMEL else {}


def rename_key(key: Any) -> Any:
    if key == "routing_algorithms":
        return "routing_algorithms"
    if isinstance(key, str):
        if "other/pk_" in key:
            return key.replace("other/pk_", "other/pk_")
        if key.startswith("ms_") and key.endswith(".yaml"):
            return "pk_" + key[3:]
        if "/ms_" in key and key.endswith(".yaml"):
            return key.replace("/ms_", "/pk_")
    return key


def apply_packing_remaps(mapping: Dict[str, Any]) -> None:
    """Rename leftover packing fields in place when they appear as siblings of strategy."""
    if not isinstance(mapping, dict):
        return
    for old, new in PACKING_FIELD_REMAP.items():
        if old in mapping and new not in mapping:
            mapping[new] = mapping.pop(old)
        elif old in mapping and new in mapping and old != new:
            mapping.pop(old)


def convert_node(node: Any) -> Any:
    if _is_single_key_map_list(node):
        out = _new_map()
        for item in node:
            key, value = next(iter(item.items()))
            out[rename_key(key)] = convert_node(value)
        apply_packing_remaps(out)
        return out
    if isinstance(node, Mapping):
        out = _new_map()
        for key, value in node.items():
            out[rename_key(key)] = convert_node(value)
        apply_packing_remaps(out)
        return out
    if isinstance(node, list):
        seq: List[Any] = CommentedSeq() if _HAS_RUAMEL else []
        for item in node:
            seq.append(convert_node(item))
        return seq
    return node


def _yaml_loader() -> Any:
    if _HAS_RUAMEL:
        yaml = cast(Any, YAML)()  # None when ruamel is not installed
        yaml.preserve_quotes = True
        yaml.width = 120
        yaml.indent(mapping=2, sequence=4, offset=2)
        return yaml
    import yaml as pyyaml  # type: ignore[import-untyped]

    class _Py:
        def load(self, handle: Any) -> Any:
            return pyyaml.safe_load(handle)

        def dump(self, data: Any, handle: Any) -> None:
            pyyaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)

    return _Py()


def convert_yaml_file(path: Path) -> bool:
    yaml = _yaml_loader()
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.load(handle)
    converted = convert_node(data)
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(converted, handle)
    return True


def convert_hpo_job(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text
    text = text.replace("other/pk_", "other/pk_")
    # Top-level SRC/ARCO pool field only; do not touch start_temp / cooling_rate.
    if path.name in {"src.json", "arco.json"}:
        data = json.loads(text)
        if "routing_algorithms" in data and "routing_algorithms" not in data:
            data["routing_algorithms"] = data.pop("constructors")
            text = json.dumps(data, indent=4) + "\n"
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def rename_packing_yamls(other_dir: Path = OTHER) -> List[Tuple[Path, Path]]:
    moved: List[Tuple[Path, Path]] = []
    for src in sorted(other_dir.glob("pk_*.yaml")):
        dest = src.with_name("pk_" + src.name[3:])
        if dest.exists():
            raise FileExistsError(dest)
        shutil.move(str(src), str(dest))
        moved.append((src, dest))
    return moved


def iter_policy_yamls(policies_dir: Path = POLICIES) -> Iterable[Path]:
    yield from sorted(policies_dir.rglob("policy_*.yaml"))
    yield from sorted((policies_dir / "other").glob("*.yaml"))


def main() -> None:
    converted = 0
    for path in iter_policy_yamls():
        convert_yaml_file(path)
        converted += 1
    moved = rename_packing_yamls()
    hpo = 0
    if HPO_JOBS.is_dir():
        for path in sorted(HPO_JOBS.glob("*.json")):
            if convert_hpo_job(path):
                hpo += 1
    print(f"converted {converted} YAML files, renamed {len(moved)} packing files, rewrote {hpo} HPO jobs")


if __name__ == "__main__":
    main()
