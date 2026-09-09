"""Pruning engine for WSmart-Route algorithm export pipeline.

Reads ``ci/export_config.json`` to enumerate every known algorithm acronym per
category, then deletes all algorithms that the caller did NOT request to keep.
Deletion is delegated to :func:`cleanup_helper.clean_by_acronym`, which already
handles YAML configs, Python dataclass configs, implementation directories /
files, and ``__init__.py`` / ``factory.py`` import cleanup.

Also handles:
- Optional-feature removal via ``--drop-features`` (META_LEARNING, HPO, EVAL,
  SECURITY, CALLBACKS, ENUMS, DATA_WEB).
- Model-specific subnet pruning: decoders/, encoders/, and factories/ within
  ``logic/src/models/subnets/`` are trimmed to only the subdirectories required
  by the kept model set (driven by ``subnet_deps`` in the config).

Usage (standalone — normally called by packager.py)::

    python logic/package/prune_codebase.py \\
        --constructors HGS ALNS BPC \\
        --selectors MS_REGULAR MS_SAVINGS \\
        --improvement RI_OROPT RI_S2OPT \\
        --acceptance AC_GD AC_OI \\
        --models AM TAM \\
        --rl-algorithms RL_REINFORCE RL_PPO \\
        --drop-features META_LEARNING HPO SECURITY

Omitting a category flag entirely means "keep everything in that category".
Passing the flag with no values means "prune the entire category".
Omitting ``--drop-features`` keeps all optional features intact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — makes cleanup_helper importable when run directly
# ---------------------------------------------------------------------------
_PACKAGES_DIR = Path(__file__).resolve().parent
# logic/package/ is 2 levels below WSmart-Route/ (previously 5 levels in src/utils/package)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # WSmart-Route/
if str(_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGES_DIR))

from cleanup_helper import clean_by_acronym  # noqa: E402  (after sys.path patch)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CONFIG_PATH = _PROJECT_ROOT / "ci" / "export_config.json"

# Maps the category keys used in the JSON to the --flag names in this CLI.
_CATEGORY_TO_FLAG: Dict[str, str] = {
    "constructor": "constructors",
    "selector": "selectors",
    "improvement": "improvement",
    "acceptance": "acceptance",
    "joint": "joint",
    "model": "models",
    "rl_algorithm": "rl_algorithms",
    "imitation_policy": "imitation_policies",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config() -> Dict:
    if not _CONFIG_PATH.exists():
        _die(f"export_config.json not found at {_CONFIG_PATH}")
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _die(message: str) -> None:
    print(f"[prune_codebase] ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _log(message: str) -> None:
    print(f"[prune_codebase] {message}")


def _collect_algorithms_by_category(config: Dict) -> Dict[str, Dict[str, str]]:
    """Return mapping: category_key → {USER_ACRONYM: internal_acronym}.

    Flattens all nested groups inside ``config["algorithms"]`` so that
    comment-only ``_comment*`` keys are silently skipped.
    """
    result: Dict[str, Dict[str, str]] = {cat: {} for cat in _CATEGORY_TO_FLAG}

    def _walk(node: Dict) -> None:
        for key, value in node.items():
            if key.startswith("_comment"):
                continue
            if not isinstance(value, dict):
                continue
            cat = value.get("category")
            internal = value.get("internal_acronym", "")
            if cat and cat in result and internal:
                result[cat][key.upper()] = internal
            else:
                # Nested group (e.g. "constructors", "selectors" wrapper objects)
                _walk(value)

    _walk(config.get("algorithms", {}))
    return result


def _build_cleanup_kwargs(category_name: str, config: Dict) -> Dict:
    """Extract the per-category cleanup kwargs from the JSON ``categories`` block."""
    cat_block = config["categories"].get(category_name)
    if not cat_block:
        _die(f"Category '{category_name}' not found in export_config.json categories block.")
    return {
        "yaml_dirs": cat_block.get("yaml_dirs", []),
        "config_dirs": cat_block.get("config_dirs", []),
        "impl_dirs": cat_block.get("impl_dirs", []),
        "yaml_prefixes": cat_block.get("yaml_prefixes"),
        "config_prefixes": cat_block.get("config_prefixes"),
    }


def _validate_acronyms(
    requested: Set[str],
    known: Set[str],
    category: str,
) -> None:
    """Warn about unknown acronyms so the caller can catch typos early."""
    unknown = requested - known
    if unknown:
        _log(
            f"WARNING: The following acronyms are not listed in export_config.json "
            f"for category '{category}' and will be ignored: " + ", ".join(sorted(unknown))
        )


def prune_optional_features(
    drop_features: List[str],
    config: Dict,
    dry_run: bool,
) -> None:
    """Run remove_*.py scripts for every feature in *drop_features*.

    Args:
        drop_features: Upper-cased feature keys to remove (e.g. ``["HPO", "SECURITY"]``).
        config: Parsed export config dict.
        dry_run: If True, log actions without executing scripts.
    """
    opt = config.get("optional_features", {})
    if not drop_features:
        return

    known_features = {k for k in opt if not k.startswith("_")}
    unknown = set(drop_features) - known_features
    if unknown:
        _log("WARNING: Unknown optional features (will be ignored): " + ", ".join(sorted(unknown)))

    for feature_key in sorted(drop_features):
        feature_key = feature_key.upper()
        if feature_key not in opt:
            continue
        feature = opt[feature_key]
        script_rel = feature.get("remove_script")
        if script_rel is None:
            _log(f"Optional feature '{feature_key}' has no remove_script — skipping (delete manually if needed).")
            continue

        script_path = _PROJECT_ROOT / script_rel
        if not script_path.exists():
            _log(f"WARNING: remove script for '{feature_key}' not found at {script_path} — skipping.")
            continue

        if dry_run:
            _log(f"  [DRY-RUN] Would run remove script for feature '{feature_key}': {script_path}")
        else:
            import subprocess  # noqa: PLC0415

            _log(f"  Dropping optional feature '{feature_key}' via {script_path.name} …")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(_PROJECT_ROOT),
                capture_output=True,
                text=True,
            )
            if result.stdout:
                for line in result.stdout.splitlines():
                    _log(f"    {line}")
            if result.returncode != 0:
                _log(f"  ERROR: {script_path.name} exited with code {result.returncode}")
                if result.stderr:
                    for line in result.stderr.splitlines():
                        _log(f"    STDERR: {line}")
            else:
                _log(f"  Feature '{feature_key}' removed successfully.")


def _get_needed_subnet_deps(
    keep_model_acronyms: List[str],
    models_cfg: Dict,
) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str]]:
    """Helper to collect union of required subnet deps across kept models."""
    needed_decoders: Set[str] = set()
    needed_encoders: Set[str] = set()
    needed_factories: Set[str] = set()
    needed_modules: Set[str] = set()
    needed_other: Set[str] = set()

    keep_upper = {a.upper() for a in keep_model_acronyms}

    for user_acr, model_entry in models_cfg.items():
        if user_acr.startswith("_comment"):
            continue
        if not isinstance(model_entry, dict):
            continue
        if user_acr.upper() not in keep_upper:
            continue
        deps = model_entry.get("subnet_deps", {})
        needed_decoders.update(deps.get("decoders", []))
        needed_encoders.update(deps.get("encoders", []))
        needed_factories.update(deps.get("factories", []))
        needed_modules.update(deps.get("modules", []))
        needed_other.update(deps.get("other", []))

    return needed_decoders, needed_encoders, needed_factories, needed_modules, needed_other


def _prune_embeddings(
    emb_cfg: Dict,
    keep_envs: Optional[List[str]],
    keep_upper: Set[str],
    emb_base: Path,
    dry_run: bool,
) -> None:
    """Helper to prune embedding files not in kept envs or kept models."""
    always_keep_emb: Set[str] = set(emb_cfg.get("always_keep", []))

    # Files to keep: always_keep + env_files for kept envs + model_files for kept models
    keep_emb_files: Set[str] = set(always_keep_emb)

    env_files_map: Dict[str, List[str]] = emb_cfg.get("env_files", {})
    keep_env_set = {e.lower() for e in (keep_envs or [])} if keep_envs is not None else None
    if keep_env_set is None:
        # No env filter — keep all env-specific files
        for files in env_files_map.values():
            keep_emb_files.update(files)
    else:
        for env_name, files in env_files_map.items():
            if env_name.lower() in keep_env_set:
                keep_emb_files.update(files)

    model_files_map: Dict[str, List[str]] = emb_cfg.get("model_files", {})
    for model_acr_lower, files in model_files_map.items():
        if model_acr_lower.upper() in keep_upper:
            keep_emb_files.update(files)

    # Build set of ALL env/model-specific files to know which to delete
    all_specific_files: Set[str] = set()
    for files in env_files_map.values():
        all_specific_files.update(files)
    for files in model_files_map.values():
        all_specific_files.update(files)

    for rel_path_str in all_specific_files:
        if rel_path_str in keep_emb_files:
            continue
        target = emb_base / rel_path_str
        if not target.exists():
            continue
        if dry_run:
            _log(f"  [DRY-RUN] Would remove embedding file: {target.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  Removing embedding file: {target.relative_to(_PROJECT_ROOT)}")
            target.unlink()


def prune_subnets(
    keep_model_acronyms: Optional[List[str]],
    keep_envs: Optional[List[str]],
    config: Dict,
    dry_run: bool,
) -> None:
    """Prune model-specific subnet subdirectories for unkept models.

    When *keep_model_acronyms* is None all models are kept → nothing pruned.
    Otherwise, computes the union of ``subnet_deps`` for the kept models and
    removes encoder/decoder/factory subdirs not in that union.  Also prunes
    modules/, other/, and embeddings/ using the same dep lists plus env info.

    Always-keep entries (``common`` for encoders/decoders, ``attention.py`` and
    ``base.py`` for factories) are never removed.
    """

    if keep_model_acronyms is None:
        _log("Subnets: no model selection provided — keeping all subnets.")
        return

    pruning_cfg = config.get("subnet_pruning", {})
    models_cfg = config.get("algorithms", {}).get("models", {})

    needed_decoders, needed_encoders, needed_factories, needed_modules, needed_other = _get_needed_subnet_deps(
        keep_model_acronyms, models_cfg
    )

    prunable_types = pruning_cfg.get("prunable_types", {})

    # --- Prune decoder subdirs ---
    decoder_cfg = prunable_types.get("decoders", {})
    decoder_base = _PROJECT_ROOT / decoder_cfg.get("base_path", "logic/src/models/subnets/decoders")
    always_keep_dec: Set[str] = set(decoder_cfg.get("always_keep", ["common"]))
    all_dec_subdirs: List[str] = decoder_cfg.get("all_subdirs", [])
    _prune_subdirs(decoder_base, all_dec_subdirs, needed_decoders | always_keep_dec, "decoder", dry_run)

    # --- Prune encoder subdirs ---
    encoder_cfg = prunable_types.get("encoders", {})
    encoder_base = _PROJECT_ROOT / encoder_cfg.get("base_path", "logic/src/models/subnets/encoders")
    always_keep_enc: Set[str] = set(encoder_cfg.get("always_keep", ["common"]))
    all_enc_subdirs: List[str] = encoder_cfg.get("all_subdirs", [])
    _prune_subdirs(encoder_base, all_enc_subdirs, needed_encoders | always_keep_enc, "encoder", dry_run)

    # --- Prune factory files ---
    factory_cfg = prunable_types.get("factories", {})
    factory_base = _PROJECT_ROOT / factory_cfg.get("base_path", "logic/src/models/subnets/factories")
    always_keep_fac: Set[str] = set(factory_cfg.get("always_keep", ["attention.py", "base.py", "__init__.py"]))
    all_fac_files: List[str] = factory_cfg.get("all_files", [])
    _prune_files(factory_base, all_fac_files, needed_factories | always_keep_fac, "factory", dry_run)

    # --- Prune modules/ files ---
    modules_cfg = prunable_types.get("modules", {})
    if modules_cfg:
        modules_base = _PROJECT_ROOT / modules_cfg.get("base_path", "logic/src/models/subnets/modules")
        always_keep_mod: Set[str] = set(modules_cfg.get("always_keep", []))
        all_mod_files: List[str] = modules_cfg.get("all_files", [])
        _prune_files(modules_base, all_mod_files, needed_modules | always_keep_mod, "module", dry_run)

    # --- Prune other/ files ---
    other_cfg = prunable_types.get("other", {})
    if other_cfg:
        other_base = _PROJECT_ROOT / other_cfg.get("base_path", "logic/src/models/subnets/other")
        always_keep_oth: Set[str] = set(other_cfg.get("always_keep", []))
        all_oth_files: List[str] = other_cfg.get("all_files", [])
        _prune_files(other_base, all_oth_files, needed_other | always_keep_oth, "other subnet", dry_run)

    # --- Prune embeddings/ (env-specific + model-specific files) ---
    emb_cfg = prunable_types.get("embeddings", {})
    if emb_cfg:
        emb_base = _PROJECT_ROOT / emb_cfg.get("base_path", "logic/src/models/subnets/embeddings")
        keep_upper = {a.upper() for a in keep_model_acronyms}
        _prune_embeddings(emb_cfg, keep_envs, keep_upper, emb_base, dry_run)


def _prune_subdirs(
    base: Path,
    all_subdirs: List[str],
    keep: Set[str],
    label: str,
    dry_run: bool,
) -> None:
    import shutil  # noqa: PLC0415

    for subdir in all_subdirs:
        if subdir in keep:
            continue
        target = base / subdir
        if not target.exists():
            continue
        if dry_run:
            _log(f"  [DRY-RUN] Would remove {label} subdir: {target.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  Removing {label} subdir: {target.relative_to(_PROJECT_ROOT)}")
            shutil.rmtree(target)


def _prune_files(
    base: Path,
    all_files: List[str],
    keep: Set[str],
    label: str,
    dry_run: bool,
) -> None:
    for fname in all_files:
        if fname in keep:
            continue
        target = base / fname
        if not target.exists():
            continue
        if dry_run:
            _log(f"  [DRY-RUN] Would remove {label} file: {target.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  Removing {label} file: {target.relative_to(_PROJECT_ROOT)}")
            target.unlink()


def _remove_constants_init_import(module_name: str) -> None:
    """Comment out a wildcard import from logic/src/constants/__init__.py."""
    import re  # noqa: PLC0415

    init_path = _PROJECT_ROOT / "logic/src/constants/__init__.py"
    if not init_path.exists():
        return
    content = init_path.read_text(encoding="utf-8")
    pattern = rf"(?m)^(from logic\.src\.constants\.{re.escape(module_name)} import \*.*)$"
    new_content = re.sub(pattern, r"# \1  # AUTO-REMOVED", content)
    if new_content != content:
        _log(f"  Updated constants/__init__.py: commented out {module_name} import")
        init_path.write_text(new_content, encoding="utf-8")


def remove_logic_dev_dirs(dry_run: bool) -> None:
    """Remove dev-only logic/ subdirectories and files not needed at runtime."""
    import shutil  # noqa: PLC0415

    targets = [
        # Dev subdirs
        _PROJECT_ROOT / "logic" / "benchmark",
        _PROJECT_ROOT / "logic" / "docs",
        _PROJECT_ROOT / "logic" / "examples",
        _PROJECT_ROOT / "logic" / "migrations",
        _PROJECT_ROOT / "logic" / "gen",
        _PROJECT_ROOT / "logic" / "store",
        _PROJECT_ROOT / "logic" / "test",
        _PROJECT_ROOT / "logic" / "src" / "py.typed",
        # Hydra tracking configs always removed (tracking module is always dropped)
        _PROJECT_ROOT / "logic" / "configs" / "tracking",
        # Batch manager config not needed in exported builds (manager subpackage removed)
        _PROJECT_ROOT / "logic" / "configs" / "batch.yaml",
        # Slurm task yaml not needed in exported builds
        _PROJECT_ROOT / "logic" / "configs" / "tasks" / "slurm.yaml",
        # Batch manager subpackage (dev/ops tooling, not part of the runtime solver)
        _PROJECT_ROOT / "logic" / "controllers" / "manager",
        # Utils subdirs that are dev/plotting/output only (not needed at runtime)
        _PROJECT_ROOT / "logic" / "src" / "utils" / "docs",
        _PROJECT_ROOT / "logic" / "src" / "utils" / "plotting",
        _PROJECT_ROOT / "logic" / "src" / "utils" / "output",
        _PROJECT_ROOT / "logic" / "src" / "utils" / "target",
        # logic/validation/ promoted from logic/src/utils/validation/
        _PROJECT_ROOT / "logic" / "validation",
        # Constants only used by removed subsystems (stats removed by remove_cli.py)
        _PROJECT_ROOT / "logic" / "src" / "constants" / "testing.py",
        _PROJECT_ROOT / "logic" / "src" / "constants" / "plotting.py",
    ]
    removed_constant_modules: List[str] = []
    for path in targets:
        if not path.exists():
            continue
        if dry_run:
            _log(f"  [DRY-RUN] Would remove: {path.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  Removing: {path.relative_to(_PROJECT_ROOT)}")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
                if path.parent.name == "constants" and path.suffix == ".py":
                    removed_constant_modules.append(path.stem)

    if not dry_run:
        for mod in removed_constant_modules:
            _remove_constants_init_import(mod)


def remove_packages_self(dry_run: bool) -> None:
    """Self-destruct: remove logic/package/ as the very last pruning step."""
    import shutil  # noqa: PLC0415

    packages_dir = _PACKAGES_DIR
    if not packages_dir.exists():
        return
    if dry_run:
        _log(f"  [DRY-RUN] Would self-remove: {packages_dir.relative_to(_PROJECT_ROOT)}")
    else:
        _log(f"  Self-removing packages dir: {packages_dir.relative_to(_PROJECT_ROOT)}")
        shutil.rmtree(packages_dir)


def _prune_all_yaml_by_prefix(
    yaml_dirs: List[str],
    yaml_prefixes: Optional[List[str]],
    dry_run: bool,
) -> None:
    """Delete ALL yaml files matching any of the prefixes (used when keeping zero algorithms)."""
    for d in yaml_dirs:
        dir_path = _PROJECT_ROOT / d
        if not dir_path.exists():
            continue
        for yaml_file in dir_path.glob("**/*.yaml"):
            name = yaml_file.stem.lower()
            should_delete = False
            if yaml_prefixes:
                for prefix in yaml_prefixes:
                    if name == prefix or name.startswith(f"{prefix}_"):
                        should_delete = True
                        break
            else:
                should_delete = True
            if should_delete:
                if dry_run:
                    _log(f"  [DRY-RUN] Would remove yaml: {yaml_file.relative_to(_PROJECT_ROOT)}")
                else:
                    _log(f"  Removing yaml: {yaml_file.relative_to(_PROJECT_ROOT)}")
                    yaml_file.unlink()


def _prune_all_configs_by_prefix(
    config_dirs: List[str],
    config_prefixes: Optional[List[str]],
    dry_run: bool,
) -> None:
    """Delete all non-__init__ .py configs (used when keeping zero algorithms in a category)."""
    for d in config_dirs:
        dir_path = _PROJECT_ROOT / d
        if not dir_path.exists():
            continue
        for py_file in dir_path.glob("**/*.py"):
            if py_file.name == "__init__.py":
                continue
            name = py_file.stem.lower()
            should_delete = False
            if config_prefixes:
                for prefix in config_prefixes:
                    if name == prefix or name.startswith(f"{prefix}_"):
                        should_delete = True
                        break
            else:
                should_delete = True
            if should_delete:
                if dry_run:
                    _log(f"  [DRY-RUN] Would remove config: {py_file.relative_to(_PROJECT_ROOT)}")
                else:
                    _log(f"  Removing config: {py_file.relative_to(_PROJECT_ROOT)}")
                    py_file.unlink()


def _prune_entire_impl_dirs(impl_dirs: List[str], dry_run: bool) -> None:
    """Remove entire implementation directories when all algorithms in a category are dropped."""
    import shutil  # noqa: PLC0415

    for d in impl_dirs:
        dir_path = _PROJECT_ROOT / d
        if not dir_path.exists():
            continue
        if dry_run:
            _log(f"  [DRY-RUN] Would remove entire impl dir: {dir_path.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  Removing entire impl dir: {dir_path.relative_to(_PROJECT_ROOT)}")
            shutil.rmtree(dir_path)


def prune_category(
    category_key: str,
    keep_acronyms: Optional[List[str]],
    all_by_category: Dict[str, Dict[str, str]],
    config: Dict,
    dry_run: bool,
) -> int:
    """Prune a single algorithm category.

    Args:
        category_key: JSON category key (e.g. ``"constructor"``).
        keep_acronyms: User-facing acronyms to RETAIN (``None`` = keep all).
        all_by_category: Pre-computed mapping of all known acronyms per category.
        config: Parsed ``export_config.json`` dict.
        dry_run: If True, log what would be deleted without actually deleting.

    Returns:
        Number of algorithms pruned.
    """
    if keep_acronyms is None:
        _log(f"Category '{category_key}': no flag provided — keeping everything.")
        return 0

    known: Dict[str, str] = all_by_category.get(category_key, {})
    keep_upper: Set[str] = {a.upper() for a in keep_acronyms}

    _validate_acronyms(keep_upper, set(known.keys()), category_key)

    to_prune: List[str] = [acr for acr in sorted(known.keys()) if acr not in keep_upper]

    if not to_prune:
        _log(f"Category '{category_key}': nothing to prune.")
        return 0

    kept_display = sorted(keep_upper & set(known.keys())) or ["(none)"]
    _log(f"Category '{category_key}': keeping {kept_display}, pruning {len(to_prune)} algorithm(s).")

    cleanup_kwargs = _build_cleanup_kwargs(category_key, config)
    pruned = 0

    # When removing ALL algorithms, use bulk deletion (faster and handles
    # yaml files whose names don't derive cleanly from internal acronyms).
    if not keep_upper:
        _log(f"  All algorithms removed — using bulk deletion for '{category_key}'.")
        _prune_all_yaml_by_prefix(
            cleanup_kwargs["yaml_dirs"],
            cleanup_kwargs.get("yaml_prefixes"),
            dry_run,
        )
        _prune_all_configs_by_prefix(
            cleanup_kwargs["config_dirs"],
            cleanup_kwargs.get("config_prefixes"),
            dry_run,
        )
        _prune_entire_impl_dirs(cleanup_kwargs["impl_dirs"], dry_run)
        return len(to_prune)

    for user_acr in to_prune:
        internal_acr = known[user_acr]
        if dry_run:
            _log(f"  [DRY-RUN] would prune {user_acr!r} (internal: {internal_acr!r})")
        else:
            _log(f"  Pruning {user_acr!r} (internal: {internal_acr!r}) …")
            clean_by_acronym(acronym=internal_acr, **cleanup_kwargs)
        pruned += 1

    return pruned


def prune_envs(keep_envs: Optional[List[str]], dry_run: bool) -> None:
    """Prune environment files not in keep_envs.

    Removes routing/*.py, tasks/*.py, and configs/envs/*.yaml for each
    environment not in the keep set.  Updates __init__.py files in affected dirs.
    """
    if keep_envs is None:
        _log("Envs: no --envs flag — keeping all environments.")
        return

    from cleanup_helper import clean_init_file  # noqa: PLC0415

    keep_set = {e.lower() for e in keep_envs}
    _log(f"Envs: keeping {sorted(keep_set) or ['(none)']}")

    impl_subdirs = [
        _PROJECT_ROOT / "logic" / "src" / "envs" / "routing",
        _PROJECT_ROOT / "logic" / "src" / "envs" / "tasks",
    ]
    yaml_dir = _PROJECT_ROOT / "logic" / "configs" / "envs"

    # Collect deletions
    to_delete = []
    affected_init_dirs: Set[Path] = set()

    for sub in impl_subdirs:
        if not sub.exists():
            continue
        affected_init_dirs.add(sub)
        for p in sub.glob("*.py"):
            if p.name in ("__init__.py", "base.py"):
                continue
            if p.stem.lower() not in keep_set:
                to_delete.append(p)

    if yaml_dir.exists():
        for p in yaml_dir.glob("*.yaml"):
            if p.stem.lower() not in keep_set:
                to_delete.append(p)

    for path in to_delete:
        if dry_run:
            _log(f"  [DRY-RUN] Would remove env file: {path.relative_to(_PROJECT_ROOT)}")
        else:
            _log(f"  Removing env file: {path.relative_to(_PROJECT_ROOT)}")
            path.unlink()

    if not dry_run:
        deleted_stems = {p.stem for p in to_delete if p.suffix == ".py"}
        for d in affected_init_dirs:
            init_file = d / "__init__.py"
            if init_file.exists():
                clean_init_file(init_file, list(deleted_stems))


def _filter_files(
    dir_path: Path,
    keep_names: List[str],
    label: str,
    dry_run: bool,
    protected_names: Optional[Set[str]] = None,
) -> None:
    """Delete .py files in dir_path whose stem doesn't match any keep_names substring."""
    from cleanup_helper import clean_init_file  # noqa: PLC0415

    if not dir_path.exists():
        return
    deleted_stems: Set[str] = set()
    for p in dir_path.glob("*.py"):
        if p.name == "__init__.py":
            continue
        if protected_names and p.name in protected_names:
            continue
        stem = p.stem.lower()

        # Match if stem equals name exactly, OR — for single-word names only (no "_") —
        # if the stem ends with "_name" (e.g. "statistical_gamma"→"gamma").
        # Multi-word names like "sim_dataset" require an exact match to avoid
        # "html_sim_dataset" being kept when only "sim_dataset" is requested.
        def _matches(s: str, name: str) -> bool:
            return s == name or ("_" not in name and s.endswith("_" + name))

        keep = any(_matches(stem, name.lower()) for name in keep_names)
        if not keep:
            deleted_stems.add(p.stem)
            if dry_run:
                _log(f"  [DRY-RUN] Would remove {label}: {p.relative_to(_PROJECT_ROOT)}")
            else:
                _log(f"  Removing {label}: {p.relative_to(_PROJECT_ROOT)}")
                p.unlink()
    if not dry_run and deleted_stems:
        init = dir_path / "__init__.py"
        if init.exists():
            clean_init_file(init, list(deleted_stems))


def prune_sim_datasets(keep_names: Optional[List[str]], dry_run: bool) -> None:
    """Filter simulation/web datasets, leaving pytorch/ untouched."""
    if keep_names is None:
        return
    _log(f"Sim datasets: keeping {keep_names or ['(none)']}")
    import shutil  # noqa: PLC0415

    datasets_dir = _PROJECT_ROOT / "logic" / "src" / "data" / "datasets"
    for sub in ("simulation", "web"):
        sub_dir = datasets_dir / sub
        if not sub_dir.exists():
            continue
        _filter_files(sub_dir, keep_names, f"{sub} dataset", dry_run)
    # Remove the web dir entirely if nothing in it was kept
    web_dir = datasets_dir / "web"
    if not dry_run and web_dir.exists():
        remaining = [p for p in web_dir.glob("*.py") if p.name != "__init__.py"]
        if not remaining:
            shutil.rmtree(web_dir)
            _log("  Removed empty web dataset dir")


def prune_distributions(keep_names: Optional[List[str]], dry_run: bool) -> None:
    """Filter distribution files."""
    if keep_names is None:
        return
    _log(f"Distributions: keeping {keep_names or ['(none)']}")
    dist_dir = _PROJECT_ROOT / "logic" / "src" / "data" / "distributions"
    _filter_files(dist_dir, keep_names, "distribution", dry_run, protected_names={"base.py"})


def prune_network(keep_names: Optional[List[str]], dry_run: bool) -> None:
    """Filter network strategy files."""
    if keep_names is None:
        return
    _log(f"Network strategies: keeping {keep_names or ['(none)']}")
    net_dir = _PROJECT_ROOT / "logic" / "src" / "data" / "network"
    _filter_files(net_dir, keep_names, "network strategy", dry_run)


def prune_policies_helpers(
    category_keeps: Dict[str, Optional[List[str]]],
    all_by_category: Dict[str, Dict[str, str]],
    config: Dict,
    dry_run: bool,
) -> None:
    """Remove files in policies/helpers/ that are not needed by any kept policy.

    Uses the ``policies_helpers_analysis`` section written by ``ci/analyze_deps.py``.
    ``__init__.py`` files are never removed.  Skips entirely if the analysis
    section is absent (run ``python ci/analyze_deps.py`` to populate it).
    """
    analysis = config.get("policies_helpers_analysis", {})
    if not analysis:
        _log(
            "  policies_helpers_analysis not found in config — skipping helpers pruning.\n"
            "  Run 'python ci/analyze_deps.py' to populate it."
        )
        return

    helpers_base = _PROJECT_ROOT / "logic" / "src" / "policies" / "helpers"
    if not helpers_base.exists():
        return

    # Policy categories that contribute helpers deps (internal keys from _CATEGORY_TO_FLAG)
    _POLICY_CAT_KEYS = ("constructor", "selector", "improvement", "acceptance")

    # Gather the kept acronyms across all policy categories
    kept_acronyms: Set[str] = set()
    for cat_key in _POLICY_CAT_KEYS:
        keep_list = category_keeps.get(cat_key)
        if keep_list is None:
            # None → flag not provided → keep ALL in this category
            kept_acronyms.update(all_by_category.get(cat_key, {}).keys())
        else:
            kept_acronyms.update(k.upper() for k in keep_list if k)

    if not kept_acronyms:
        _log("  No policies kept — removing all non-init helpers files.")

    # Build union of needed helpers files from the analysis
    needed: Set[str] = set()
    unanalyzed: List[str] = []
    for acr in kept_acronyms:
        if acr in analysis:
            needed.update(analysis[acr])
        else:
            unanalyzed.append(acr)

    if unanalyzed:
        _log(
            f"  WARNING: {len(unanalyzed)} kept policies have no helpers analysis entry.\n"
            f"  Run 'python ci/analyze_deps.py' to refresh, then re-run the packager.\n"
            f"  Skipping helpers pruning to avoid breaking un-analyzed policies."
        )
        return

    # Remove .py files in helpers/ not in needed (__init__.py always kept)
    all_helpers_files = sorted(
        p for p in helpers_base.rglob("*.py") if "__pycache__" not in p.parts and not p.name.startswith("__")
    )

    removed = 0
    for f in all_helpers_files:
        rel = f.relative_to(helpers_base).as_posix()
        if rel not in needed:
            if dry_run:
                _log(f"  [DRY-RUN] Would remove: {f.relative_to(_PROJECT_ROOT)}")
            else:
                _log(f"  Removing: {f.relative_to(_PROJECT_ROOT)}")
                f.unlink()
                removed += 1

    if not dry_run:
        _log(f"  Removed {removed} unused helpers files.")


def prune_empty_dirs(dry_run: bool) -> None:
    """Remove directories that have become effectively empty after pruning."""
    from cleanup_helper import remove_empty_dirs  # noqa: PLC0415

    scan_roots = [
        _PROJECT_ROOT / "logic" / "src" / "policies",
        _PROJECT_ROOT / "logic" / "src" / "models",
        _PROJECT_ROOT / "logic" / "src" / "pipeline",
        _PROJECT_ROOT / "logic" / "src" / "data",
        _PROJECT_ROOT / "logic" / "src" / "envs",
        _PROJECT_ROOT / "logic" / "configs",
    ]

    existing = [p for p in scan_roots if p.exists()]
    if not existing:
        return

    if dry_run:
        _log("  [DRY-RUN] Would prune effectively-empty directories.")
        return

    _log("  Pruning effectively-empty directories …")
    remove_empty_dirs(_PROJECT_ROOT, existing)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prune_codebase.py",
        description=(
            "Prune unselected algorithms from the WSmart-Route codebase. "
            "Each --<category> flag accepts a space-separated list of "
            "user-facing acronyms to KEEP. Omitting a flag keeps the entire "
            "category intact. Passing the flag with no values prunes the whole "
            "category."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--constructors",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Route construction algorithms to keep (e.g. HGS ALNS BPC).",
    )
    p.add_argument(
        "--selectors",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Mandatory selection policies to keep (e.g. MS_REGULAR MS_PPK).",
    )
    p.add_argument(
        "--improvement",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Route improvement operators to keep (e.g. RI_OROPT RI_S2OPT).",
    )
    p.add_argument(
        "--acceptance",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Acceptance criteria to keep (e.g. AC_GD AC_OI AC_SA).",
    )
    p.add_argument(
        "--joint",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Joint selection-and-construction methods to keep (e.g. JGO JSA).",
    )
    p.add_argument(
        "--models",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        help="Neural model architectures to keep (e.g. AM TAM MDAM).",
    )
    p.add_argument(
        "--rl-algorithms",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        dest="rl_algorithms",
        help="RL training algorithms to keep (e.g. RL_REINFORCE RL_PPO).",
    )
    p.add_argument(
        "--imitation-policies",
        nargs="*",
        metavar="ACRONYM",
        default=None,
        dest="imitation_policies",
        help="Imitation learning policy types to keep (e.g. IP_HGS IP_ALNS).",
    )
    p.add_argument(
        "--drop-features",
        nargs="*",
        metavar="FEATURE",
        default=None,
        dest="drop_features",
        help=(
            "Optional pipeline features to REMOVE (e.g. META_LEARNING HPO SECURITY). "
            "Available: META_LEARNING, HPO, EVAL, SECURITY, CALLBACKS, ENUMS, "
            "DATA_WEB, TRACKING, CLI. Omitting this flag keeps all features."
        ),
    )
    p.add_argument(
        "--envs",
        nargs="*",
        metavar="ENV",
        default=None,
        help="Environment names to KEEP (e.g. vrpp). Others are removed. Omit to keep all.",
    )
    p.add_argument(
        "--sim-datasets",
        nargs="*",
        metavar="NAME",
        default=None,
        dest="sim_datasets",
        help="Simulation dataset stems to KEEP (e.g. gen_dataset sim_dataset). Pytorch left untouched.",
    )
    p.add_argument(
        "--distributions",
        nargs="*",
        metavar="NAME",
        default=None,
        help="Distribution stems to KEEP (e.g. gamma empirical).",
    )
    p.add_argument(
        "--network",
        nargs="*",
        metavar="NAME",
        default=None,
        help="Network strategy stems to KEEP (e.g. file).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be deleted without actually deleting anything.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        metavar="PATH",
        help=f"Path to export_config.json (default: {_CONFIG_PATH}).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path: Path = args.config
    if not config_path.exists():
        _die(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as fh:
        config = json.load(fh)

    all_by_category = _collect_algorithms_by_category(config)

    if args.dry_run:
        _log("=== DRY RUN — no files will be deleted ===")

    # Map each category key → the user-provided keep list (None if flag omitted)
    category_keeps: Dict[str, Optional[List[str]]] = {
        "constructor": args.constructors,
        "selector": args.selectors,
        "improvement": args.improvement,
        "acceptance": args.acceptance,
        "joint": args.joint,
        "model": args.models,
        "rl_algorithm": args.rl_algorithms,
        "imitation_policy": args.imitation_policies,
    }

    total_pruned = 0
    for cat_key, keep_list in category_keeps.items():
        total_pruned += prune_category(
            category_key=cat_key,
            keep_acronyms=keep_list,
            all_by_category=all_by_category,
            config=config,
            dry_run=args.dry_run,
        )

    # Prune policies/helpers/ based on pre-computed per-policy dep analysis.
    _log("Pruning policies/helpers/ …")
    prune_policies_helpers(
        category_keeps=category_keeps,
        all_by_category=all_by_category,
        config=config,
        dry_run=args.dry_run,
    )

    # Prune model-specific subnets after the model category is decided.
    _log("Pruning model subnets …")
    prune_subnets(
        keep_model_acronyms=category_keeps.get("model"),
        keep_envs=args.envs,
        config=config,
        dry_run=args.dry_run,
    )

    # Prune environments.
    _log("Pruning environments …")
    prune_envs(keep_envs=args.envs, dry_run=args.dry_run)

    # Drop optional pipeline features on request.
    drop_features: List[str] = [f.upper() for f in (args.drop_features or [])]
    if drop_features:
        _log(f"Dropping optional features: {drop_features}")
        prune_optional_features(drop_features, config, dry_run=args.dry_run)
    else:
        _log("No optional features dropped (use --drop-features to remove META_LEARNING, HPO, etc.).")

    # Filter data/distributions/network (runs before self-destruct so scripts are still available).
    if args.sim_datasets is not None:
        _log("Filtering simulation datasets …")
        prune_sim_datasets(keep_names=args.sim_datasets, dry_run=args.dry_run)
    if args.distributions is not None:
        _log("Filtering distributions …")
        prune_distributions(keep_names=args.distributions, dry_run=args.dry_run)
    if args.network is not None:
        _log("Filtering network strategies …")
        prune_network(keep_names=args.network, dry_run=args.dry_run)

    # Remove dev-only logic/ subdirs (benchmark, docs, examples, test, utils/plotting, …).
    _log("Removing dev-only logic/ subdirectories …")
    remove_logic_dev_dirs(dry_run=args.dry_run)

    # Clean up directories that became effectively empty after all the above deletions.
    _log("Pruning effectively-empty directories …")
    prune_empty_dirs(dry_run=args.dry_run)

    # Self-destruct: remove this packages dir last so all remove_*.py scripts are
    # available for the entire pruning run.
    _log("Removing logic/package/ (self-cleanup) …")
    remove_packages_self(dry_run=args.dry_run)

    _log(f"{'[DRY-RUN] ' if args.dry_run else ''}Done. Total algorithms pruned: {total_pruned}.")


if __name__ == "__main__":
    main()
