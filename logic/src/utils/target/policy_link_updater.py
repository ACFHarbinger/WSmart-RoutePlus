"""Policy YAML config-override updater.

One parameterised implementation of the batch updater that rewrites the
``mandatory_selection`` or ``route_improvement`` inline override in policy
YAML files for one or more route constructors. The two public sides keep
their historical names:

- ``update_mandatory_selection`` / ``list_available_ms_strategies`` /
  ``list_strategy_keys`` (files prefixed ``ms_``);
- ``update_route_improvement`` / ``list_available_ri_improvers`` /
  ``list_improver_keys`` (files prefixed ``ri_``).

Quick start::

    from logic.src.utils.target.policy_link_updater import (
        update_mandatory_selection,
        update_route_improvement,
    )

    update_mandatory_selection(
        constructors=["aco_hh", "alns", "bpc"],
        ms_yaml="ms_service_level",
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
from typing import List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIGS_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, "../../../configs/policies"))


def _resolve_stem(yaml_arg: str) -> str:
    """Normalize a yaml filename argument to a bare stem.

    Args:
        yaml_arg: e.g. ``"ms_service_level"``, ``"ms_service_level.yaml"``,
            or ``"other/ms_service_level.yaml"``.

    Returns:
        Bare stem, e.g. ``"ms_service_level"``.
    """
    name = os.path.basename(yaml_arg)
    if name.endswith(".yaml"):
        name = name[:-5]
    return name


def _field_re(field_label: str) -> re.Pattern:
    return re.compile(rf"({field_label}:\s*)\{{[^}}]+\}}")


def _list_available(configs_dir: str, file_prefix: str) -> List[str]:
    """Return the sorted stems of the ``<prefix>*.yaml`` strategy files."""
    other_dir = os.path.join(configs_dir, "other")
    if not os.path.isdir(other_dir):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(other_dir)
        if f.startswith(file_prefix) and f.endswith(".yaml")
    )


def _list_keys(yaml_stem: str, configs_dir: str) -> List[str]:
    """Return the top-level keys defined in the strategy file *yaml_stem*."""
    path = os.path.join(configs_dir, "other", f"{yaml_stem}.yaml")
    if not os.path.isfile(path):
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
    *constructors* with an inline dict referencing the strategy file and
    *keys*.

    Returns:
        List of ``(filepath, replacement_count)`` tuples for each modified file.

    Raises:
        FileNotFoundError: If the strategy file does not exist.
        ValueError: If a requested key is not present in the strategy file.
    """
    stem = _resolve_stem(yaml_stem_arg)
    strategy_path = os.path.join(configs_dir, "other", f"{stem}.yaml")
    if not os.path.isfile(strategy_path):
        raise FileNotFoundError(
            f"{missing_file_noun} not found: {strategy_path}\n"
            f"Available: {_list_available(configs_dir, file_prefix)}"
        )

    available_keys = _list_keys(stem, configs_dir)
    for key in keys:
        if key not in available_keys:
            raise ValueError(f"Key '{key}' not found in {stem}.yaml. Available keys: {available_keys}")

    keys_str = ", ".join(f'"{k}"' for k in keys)
    replacement = f'\\g<1>{{ "other/{stem}.yaml": [{keys_str}] }}'

    modified: List[Tuple[str, int]] = []
    pattern = _field_re(field_label)

    for constructor in constructors:
        policy_file = os.path.join(configs_dir, f"policy_{constructor}.yaml")
        if not os.path.isfile(policy_file):
            if verbose:
                print(f"  [WARN] Policy file not found: {policy_file}")
            continue

        with open(policy_file, "r", encoding="utf-8") as fh:
            content = fh.read()

        new_content, count = pattern.subn(replacement, content)

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


# --- mandatory selection (files prefixed ``ms_``) ----------------------------


def list_available_ms_strategies(configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return the sorted stems of the available mandatory-selection files."""
    return _list_available(configs_dir, "ms_")


def list_strategy_keys(ms_yaml: str, configs_dir: str = _DEFAULT_CONFIGS_DIR) -> List[str]:
    """Return the top-level keys defined in *ms_yaml*."""
    return _list_keys(_resolve_stem(ms_yaml), configs_dir)


def update_mandatory_selection(
    constructors: List[str],
    ms_yaml: str,
    keys: List[str],
    configs_dir: str = _DEFAULT_CONFIGS_DIR,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Tuple[str, int]]:
    """Update ``mandatory_selection`` in the policy YAML files of *constructors*."""
    return _update(
        field_label="mandatory_selection",
        file_prefix="ms_",
        constructors=constructors,
        yaml_stem_arg=ms_yaml,
        keys=keys,
        configs_dir=configs_dir,
        dry_run=dry_run,
        verbose=verbose,
        missing_file_noun="Mandatory-selection file",
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
    """Update ``route_improvement`` in the policy YAML files of *constructors*."""
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
