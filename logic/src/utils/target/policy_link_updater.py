"""Policy YAML config-override updater.

One parameterised implementation of the batch updater that rewrites the
``packing`` or ``route_improvement`` inline override in policy
YAML files for one or more routing algorithms. The two public sides keep
their historical names:

- ``update_packing`` / ``list_available_packing_strategies`` /
  ``list_strategy_keys`` (files prefixed ``pk_``);
- ``update_route_improvement`` / ``list_available_ri_improvers`` /
  ``list_improver_keys`` (files prefixed ``ri_``).

Quick start::

    from logic.src.utils.target.policy_link_updater import (
        update_packing,
        update_route_improvement,
    )

    update_packing(
        constructors=["aco_hh", "alns", "bpc"],
        pk_yaml="pk_service_level",
        keys=["service_level1", "service_level2"],
    )

    update_route_improvement(
        constructors=["aco_hh", "alns", "bpc"],
        ri_yaml="ri_ftsp",
        keys=["ftsp"],
    )
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

from logic.src.utils.configs.policy_files import find_policy_file

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIGS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../configs/policies"))


def _resolve_stem(yaml_arg: str) -> str:
    """Normalize a yaml filename argument to a bare stem.

    Args:
        yaml_arg: e.g. ``"ms_service_level"``, ``"ms_service_level.yaml"``,
            or ``"other/packing/pk_service_level.yaml"``.

    Returns:
        Bare stem, e.g. ``"ms_service_level"``.
    """
    name = os.path.basename(yaml_arg)
    if name.endswith(".yaml"):
        name = name[:-5]
    return name


# Stage files live in other/<stage>/ by their prefix.
_OTHER_SUBDIRS = {"pk_": "packing", "ri_": "route_improvement", "ac_": "acceptance_criteria"}


def _other_rel(stem: str) -> str:
    """Path of a stage file relative to the policies folder, e.g. ``other/packing/pk_regular.yaml``."""
    subdir = next((d for p, d in _OTHER_SUBDIRS.items() if stem.startswith(p)), None)
    return f"other/{subdir}/{stem}.yaml" if subdir else f"other/{stem}.yaml"


def _field_re(field_label: str) -> re.Pattern:
    return re.compile(rf"({field_label}:\s*)\{{[^}}]+\}}")


def _rewrite_field(content: str, field_label: str, stem: str, keys: List[str]) -> Tuple[str, int]:
    """Replace an inline or block ``field_label`` section.

    Packaged policy files use a nested block, not ``packing: { ... }``.
    """
    keys_str = ", ".join(f'"{key}"' for key in keys)
    inline = f'{{ "{_other_rel(stem)}": [{keys_str}] }}'
    new_content, count = _field_re(field_label).subn(rf"\g<1>{inline}", content)
    if count:
        return new_content, count
    block = re.compile(rf"(?m)^([ \t]*){re.escape(field_label)}:[ \t]*\n(?:\1[ \t]+.*(?:\n|$))*")

    def _repl(match: re.Match) -> str:
        indent = match.group(1)
        return f"{indent}{field_label}: {inline}\n"

    return block.subn(_repl, content)


def _list_available(configs_dir: str, file_prefix: str) -> List[str]:
    """Return the sorted stems of the ``<prefix>*.yaml`` strategy files."""
    other_dir = Path(configs_dir) / "other"
    if not other_dir.is_dir():
        return []
    return sorted(p.stem for p in other_dir.rglob(f"{file_prefix}*.yaml"))


def _list_keys(yaml_stem: str, configs_dir: str) -> List[str]:
    """Return the top-level keys defined in the strategy file *yaml_stem*."""
    path = find_policy_file(f"{yaml_stem}.yaml", configs_dir)
    if path is None:
        return []
    keys: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip()
            if stripped and not stripped.startswith(" ") and not stripped.startswith("#"):
                m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", stripped)
                if m:
                    keys.append(m.group(1))
    return keys


def _update(
    field_label: str,
    file_prefix: str,
    constructors: List[str],
    yaml_stem_arg: str,
    keys: List[str],
    configs_dir: str,
    dry_run: bool,
    verbose: bool,
    missing_file_noun: str,
) -> List[Tuple[str, int]]:
    """Replace every ``<field_label>: { ... }`` entry in the policy files of
    *routing algorithms* with an inline dict referencing the strategy file and
    *keys*.

    Returns:
        List of ``(filepath, replacement_count)`` tuples for each modified file.

    Raises:
        FileNotFoundError: If the strategy file does not exist.
        ValueError: If a requested key is not present in the strategy file.
    """
    stem = _resolve_stem(yaml_stem_arg)
    strategy_path = os.path.join(configs_dir, _other_rel(stem))
    if find_policy_file(f"{stem}.yaml", configs_dir) is None:
        raise FileNotFoundError(
            f"{missing_file_noun} not found: {strategy_path}\n"
            f"Available: {_list_available(configs_dir, file_prefix)}"
        )

    available_keys = _list_keys(stem, configs_dir)
    for key in keys:
        if key not in available_keys:
            raise ValueError(f"Key '{key}' not found in {stem}.yaml. Available keys: {available_keys}")

    modified: List[Tuple[str, int]] = []

    for constructor in constructors:
        found = find_policy_file(f"policy_{constructor}.yaml", configs_dir)
        policy_file = str(found) if found is not None else os.path.join(configs_dir, f"policy_{constructor}.yaml")
        if found is None:
            if verbose:
                print(f"  [WARN] Policy file not found: {policy_file}")
            continue

        with open(policy_file, "r", encoding="utf-8") as fh:
            content = fh.read()

        new_content, count = _rewrite_field(content, field_label, stem, keys)

        if count == 0:
            if verbose:
                print(f"  [SKIP] No {field_label} {{...}} found in policy_{constructor}.yaml")
            continue

        modified.append((policy_file, count))

        if verbose:
            prefix = "[DRY RUN]" if dry_run else "[UPDATED]"
            print(f"  {prefix} policy_{constructor}.yaml — {count} occurrence(s) → {stem}: {keys}")

        if not dry_run:
            with open(policy_file, "w", encoding="utf-8") as fh:
                fh.write(new_content)

    if verbose:
        status = "Would update" if dry_run else "Updated"
        print(f"\n{status} {len(modified)} policy file(s).")

    return modified


# --- packing (files prefixed ``pk_``) ----------------------------


def list_available_packing_strategies(configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return the sorted stems of the available packing files."""
    return _list_available(configs_dir, "pk_")


def list_strategy_keys(pk_yaml: str, configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return the top-level keys defined in *pk_yaml*."""
    return _list_keys(_resolve_stem(pk_yaml), configs_dir)


def update_packing(
    constructors: List[str],
    pk_yaml: str,
    keys: List[str],
    configs_dir: str = _DEFAULT_CONFIGS_DIR,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Tuple[str, int]]:
    """Update ``packing`` in the policy YAML files of *routing algorithms*."""
    return _update(
        field_label="packing",
        file_prefix="pk_",
        constructors=constructors,
        yaml_stem_arg=pk_yaml,
        keys=keys,
        configs_dir=configs_dir,
        dry_run=dry_run,
        verbose=verbose,
        missing_file_noun="Packing file",
    )


# --- route improvement (files prefixed ``ri_``) ------------------------------


def list_available_ri_improvers(configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return the sorted stems of the available route-improver files."""
    return _list_available(configs_dir, "ri_")


def list_improver_keys(ri_yaml: str, configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return the top-level keys defined in *ri_yaml*."""
    return _list_keys(_resolve_stem(ri_yaml), configs_dir)


def update_route_improvement(
    constructors: List[str],
    ri_yaml: str,
    keys: List[str],
    configs_dir: str = _DEFAULT_CONFIGS_DIR,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Tuple[str, int]]:
    """Update ``route_improvement`` in the policy YAML files of *routing algorithms*."""
    return _update(
        field_label="route_improvement",
        file_prefix="ri_",
        constructors=constructors,
        yaml_stem_arg=ri_yaml,
        keys=keys,
        configs_dir=configs_dir,
        dry_run=dry_run,
        verbose=verbose,
        missing_file_noun="Route-improver file",
    )
