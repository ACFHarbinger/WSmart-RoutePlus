"""Helper logic for cleaning up components (configs, overrides, implementations, imports)."""

import re
import shutil
from pathlib import Path
from typing import Optional

# Set of directory names and files that should not be deleted under any circumstances
PROTECTED_DIRS = {
    "routing",
    "exact_and_decomposition_solvers",
    "hyper_heuristics",
    "learning_algorithms",
    "learning_heuristic_algorithms",
    "learning_matheuristic_algorithms",
    "matheuristics",
    "meta_heuristics",
    "other_algorithms",
    "acceptance_criteria",
    "packing",
    "route_improvement",
    "packing_and_routing",
    "base",
    "common",
    "generators",
    "routing",
    "tasks",
    "envs",
    "models",
    "policies",
    "subnets",
    "core",
    "meta",
    "rl",
    "hpo",
    "losses",
}

# Category 1: Routing Algorithms
ROUTING_ALGORITHMS = {
    "yaml_dirs": ["logic/configs/policies"],
    "config_dirs": ["logic/src/configs/policies"],
    "impl_dirs": ["logic/src/policies/routing"],
    "yaml_prefixes": ["policy"],
    "config_prefixes": [],
}

# Category 2: Others (packing, route improver, acceptance criteria, packing and routing)
POLICY_OTHERS = {
    "yaml_dirs": ["logic/configs/policies/other"],
    "config_dirs": ["logic/src/configs/policies/other"],
    "impl_dirs": [
        "logic/src/policies/acceptance_criteria",
        "logic/src/policies/packing",
        "logic/src/policies/route_improvement",
        "logic/src/policies/packing_and_routing",
    ],
    "yaml_prefixes": ["ac", "ms", "ri"],
    "config_prefixes": ["ac", "ms", "ri"],
}

# Category 3: Envs
ENVS = {
    "yaml_dirs": ["logic/configs/envs"],
    "config_dirs": ["logic/src/configs/envs"],
    "impl_dirs": ["logic/src/envs"],
    "yaml_prefixes": [],
    "config_prefixes": [],
}

# Category 4: Models
MODELS = {
    "yaml_dirs": ["logic/configs/models"],
    "config_dirs": ["logic/src/configs/models"],
    "impl_dirs": ["logic/src/models"],
    "yaml_prefixes": [],
    "config_prefixes": [],
}

# Category 5: RL Algorithms
RL_ALGORITHMS = {
    "yaml_dirs": [],
    "config_dirs": ["logic/src/configs/rl/core"],
    "impl_dirs": ["logic/src/pipeline/rl"],
    "yaml_prefixes": [],
    "config_prefixes": [],
}

# Category 6: Imitation Policies
IMITATION_POLICIES = {
    "yaml_dirs": ["logic/configs/models/policies"],
    "config_dirs": ["logic/src/configs/rl/policies"],
    "impl_dirs": ["logic/src/models/policies"],
    "yaml_prefixes": [],
    "config_prefixes": [],
}


def get_project_root() -> Path:
    """Find WSmart-Route root directory."""
    return Path(__file__).resolve().parents[2]


def clean_init_file(init_file_path: Path, deleted_stems: list):
    """Comment out imports of deleted components in __init__.py files."""
    if not init_file_path.exists():
        return
    try:
        lines = init_file_path.read_text(errors="ignore").splitlines()
        new_lines = []
        modified = False
        for line in lines:
            should_exclude = False
            for stem in deleted_stems:
                # Relative import: from .stem / from .stem_policy / from .selection_stem
                pattern = rf"^\s*from\s+\.({re.escape(stem)}|{re.escape(stem)}_policy|selection_{re.escape(stem)})\b"
                if re.search(pattern, line):
                    should_exclude = True
                    break
                # Absolute import ending with .stem (e.g. from logic.src.pipeline.rl.core.ppo import ...)
                pattern_abs = rf"^\s*from\s+[a-zA-Z_][a-zA-Z0-9_.]*\.{re.escape(stem)}\s+import"
                if re.search(pattern_abs, line):
                    should_exclude = True
                    break
                pattern2 = rf"^\s*import\s+{re.escape(stem)}\b"
                if re.search(pattern2, line):
                    should_exclude = True
                    break

            if should_exclude:
                print(f"Commenting out import in {init_file_path.relative_to(get_project_root())}: {line.strip()}")
                new_lines.append(f"# {line}  # AUTO-CLEANED")
                modified = True
                continue

            cleaned_line = line
            for stem in deleted_stems:
                pattern_dict = rf'^\s*["\']({re.escape(stem)}|{re.escape(stem).upper()}Config|Vectorized{re.escape(stem).upper()}Policy|{re.escape(stem).upper()}Policy|{re.escape(stem)}_policy)["\']\s*:'
                pattern_all = rf'^\s*["\']({re.escape(stem).upper()}Config|Vectorized{re.escape(stem).upper()}Policy|{re.escape(stem).upper()}Policy|{re.escape(stem)}_policy)["\']\s*,\s*$'
                if re.search(pattern_dict, line) or re.search(pattern_all, line):
                    cleaned_line = f"# {line}  # AUTO-CLEANED"
                    print(f"Commenting out item in {init_file_path.relative_to(get_project_root())}: {line.strip()}")
                    modified = True
                    break

            new_lines.append(cleaned_line)

        if modified:
            init_file_path.write_text("\n".join(new_lines) + "\n")
    except Exception as e:
        print(f"Error cleaning {init_file_path}: {e}")


def clean_factory_file(factory_file_path: Path, deleted_stems: list):
    """Comment out module registration strings in factory.py files."""
    if not factory_file_path.exists():
        return
    try:
        lines = factory_file_path.read_text(errors="ignore").splitlines()
        new_lines = []
        modified = False
        for line in lines:
            cleaned_line = line
            for stem in deleted_stems:
                pattern = rf'^\s*["\']({re.escape(stem)})["\']\s*,\s*$'
                if re.search(pattern, line):
                    cleaned_line = f"# {line}  # AUTO-CLEANED"
                    print(
                        f"Commenting out module entry in {factory_file_path.relative_to(get_project_root())}: {line.strip()}"
                    )
                    modified = True
                    break
            new_lines.append(cleaned_line)
        if modified:
            factory_file_path.write_text("\n".join(new_lines) + "\n")
    except Exception as e:
        print(f"Error cleaning {factory_file_path}: {e}")


def remove_path(path: Path):
    """Delete a file or directory safely."""
    if not path.exists():
        return
    root = get_project_root()
    rel_path = path.relative_to(root) if path.is_absolute() else path
    if path.is_dir():
        print(f"Removing directory: {rel_path}")
        shutil.rmtree(path)
    else:
        print(f"Removing file: {rel_path}")
        path.unlink()


def _match_acronym(name_lower: str, acronym: str) -> bool:
    """Helper to check if a name matches the acronym."""
    if name_lower in (acronym, f"policy_{acronym}", f"selection_{acronym}"):
        return True
    if name_lower.endswith(f"_{acronym}") or name_lower.startswith(f"{acronym}_") or f"_{acronym}_" in name_lower:
        return True

    # Forward: initials of name_lower == acronym
    parts = name_lower.split("_")
    if len(parts) > 1:
        stop_words = {"and", "or", "with", "for", "the", "of", "to", "in", "on", "at", "by", "from"}
        initials = "".join([part[0] for part in parts if part and part not in stop_words])
        if initials == acronym:
            return True

    # Reverse: check if name_lower (or its suffix) equals the initials of the acronym.
    # This handles e.g. "alns" matching "adaptive_large_neighborhood_search" or
    # "ac_abm" matching "adaptive_boltzmann_metropolis".
    acronym_parts = [p for p in acronym.split("_") if p]
    if len(acronym_parts) > 1:
        acronym_initials = "".join(p[0] for p in acronym_parts)
        if (
            name_lower == acronym_initials
            or name_lower.endswith(f"_{acronym_initials}")
            or name_lower.startswith(f"{acronym_initials}_")
        ):
            return True

    return False


def _find_yaml_to_delete(
    yaml_dirs: list, acronym: Optional[str], yaml_prefixes: Optional[list], root: Path, to_delete: set
):
    """Scan and find YAML configs to delete."""
    if acronym is None:
        return
    for d in yaml_dirs:
        dir_path = root / d
        if not dir_path.exists():
            continue
        for p in dir_path.glob("**/*.yaml"):
            name_lower = p.stem.lower()
            matched = _match_acronym(name_lower, acronym)
            if not matched and yaml_prefixes:
                for prefix in yaml_prefixes:
                    if name_lower == f"{prefix}_{acronym}":
                        matched = True
                        break
            if matched:
                to_delete.add(p)


def _find_configs_to_delete(
    config_dirs: list,
    acronym: Optional[str],
    config_prefixes: Optional[list],
    root: Path,
    to_delete: set,
    deleted_stems: set,
    affected_init_dirs: set,
):
    """Scan and find Python configs to delete."""
    if acronym is None:
        return
    for d in config_dirs:
        dir_path = root / d
        if not dir_path.exists():
            continue
        affected_init_dirs.add(dir_path)
        for p in dir_path.glob("**/*.py"):
            name_lower = p.stem.lower()
            if name_lower == "__init__":
                continue
            matched = _match_acronym(name_lower, acronym)
            if not matched and config_prefixes:
                for prefix in config_prefixes:
                    if name_lower == f"{prefix}_{acronym}":
                        matched = True
                        break
            if matched:
                to_delete.add(p)
                deleted_stems.add(p.stem)


def _find_impls_to_delete(
    impl_dirs: list,
    acronym: str,
    root: Path,
    to_delete: set,
    deleted_stems: set,
    affected_init_dirs: set,
    affected_factory_dirs: set,
):
    """Scan and find implementations to delete."""
    for d in impl_dirs:
        dir_path = root / d
        if not dir_path.exists():
            continue
        affected_init_dirs.add(dir_path)
        affected_factory_dirs.add(dir_path)

        for p in dir_path.glob("**/*"):
            if p.name in ("__init__.py", "factory.py", "registry.py"):
                continue

            if p.is_file() and p.suffix == ".py":
                name_lower = p.stem.lower()
                matched = _match_acronym(name_lower, acronym)

                if not matched:
                    try:
                        content = p.read_text(errors="ignore")
                        pattern = rf'register\(\s*["\']{re.escape(acronym)}["\']\s*\)'
                        if re.search(pattern, content):
                            matched = True
                    except Exception:
                        pass

                if matched:
                    deleted_stems.add(p.stem)
                    parent = p.parent
                    if parent.name in PROTECTED_DIRS:
                        to_delete.add(p)
                    else:
                        to_delete.add(parent)
                        # Also register the directory name so parent __init__.py
                        # imports like `from .directory_name import ...` get cleaned.
                        deleted_stems.add(parent.name)

            elif p.is_dir():
                name_lower = p.name.lower()
                matched = _match_acronym(name_lower, acronym)
                if matched and p.name not in PROTECTED_DIRS:
                    to_delete.add(p)
                    deleted_stems.add(p.name)


_SKIP_DIR_NAMES: set = {".venv", "venv", ".git", "__pycache__", "node_modules", ".mypy_cache", ".pytest_cache"}


def _is_effectively_empty(path: Path) -> bool:
    """Return True if directory only contains __init__.py and/or effectively-empty subdirs."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.name in _SKIP_DIR_NAMES:
            continue
        if child.is_file():
            if child.name != "__init__.py":
                return False
        elif child.is_dir():  # noqa: SIM102
            if not _is_effectively_empty(child):
                return False
    return True


def remove_empty_dirs(root: Path, scan_paths: list) -> None:
    """Remove dirs that only contain __init__.py (and/or other empty subdirs).

    Iterates repeatedly until no further removals occur (handles nested emptiness).
    """
    changed = True
    while changed:
        changed = False
        for scan_path in scan_paths:
            if not scan_path.exists() or not scan_path.is_dir():
                continue
            # Collect all subdirs, deepest first — skip non-project dirs
            subdirs = sorted(
                [
                    p for p in scan_path.rglob("*")
                    if p.is_dir() and not any(part in _SKIP_DIR_NAMES for part in p.parts)
                ],
                key=lambda p: len(p.parts),
                reverse=True,
            )
            for dirpath in subdirs:
                if dirpath == scan_path:
                    continue
                if _is_effectively_empty(dirpath):
                    print(f"Removing effectively empty dir: {dirpath.relative_to(root)}")
                    shutil.rmtree(dirpath)
                    changed = True


def clean_by_acronym(
    acronym: str,
    yaml_dirs: list,
    config_dirs: list,
    impl_dirs: list,
    yaml_prefixes: Optional[list] = None,
    config_prefixes: Optional[list] = None,
):
    """Find and delete configs, overrides, implementations and clean imports for a given acronym."""
    acronym = acronym.lower().strip()
    if acronym.startswith("acronyms="):
        acronym = acronym.split("=", 1)[1].strip()
    if not acronym:
        return

    root = get_project_root()
    to_delete = set()
    deleted_stems = set()
    affected_init_dirs = set()
    affected_factory_dirs = set()

    _find_yaml_to_delete(yaml_dirs, acronym, yaml_prefixes, root, to_delete)
    _find_configs_to_delete(config_dirs, acronym, config_prefixes, root, to_delete, deleted_stems, affected_init_dirs)
    _find_impls_to_delete(impl_dirs, acronym, root, to_delete, deleted_stems, affected_init_dirs, affected_factory_dirs)

    if not to_delete:
        print(f"No component files/directories found matching acronym: {acronym}")
        return

    for path in sorted(list(to_delete), key=lambda x: len(str(x)), reverse=True):
        remove_path(path)

    stems_list = list(deleted_stems) + [acronym]
    for d in affected_init_dirs:
        for init_file in d.glob("**/__init__.py"):
            clean_init_file(init_file, stems_list)

    for d in affected_factory_dirs:
        for factory_file in d.glob("**/factory.py"):
            clean_factory_file(factory_file, stems_list)
