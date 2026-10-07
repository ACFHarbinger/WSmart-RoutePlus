"""Static dependency analyzer for WSmart-Route model subnet and policy helpers pruning.

Performs AST-based import analysis to discover:

1. **Model subnet dependencies** — which files in
   ``logic/src/models/subnets/{modules,other,factories,decoders,encoders}``
   and ``logic/src/models/subnets/embeddings/`` are actually imported by each
   model in ``logic/src/models/core/<model>/``.

2. **Policy helpers dependencies** — which files in
   ``logic/src/policies/helpers/`` (including 2nd-order transitive deps)
   are required by each policy in the constructors, improvers,
   mandatory_selections, acceptance_criteria, and selection_and_construction
   categories.

Results are written back to ``ci/export_config.json``:
- Model deps → each model's ``subnet_deps`` key
- Embeddings → ``subnet_pruning.prunable_types.embeddings.{env_files,model_files}``
- Policy helpers → ``policies_helpers_analysis`` (per-policy dep lists)

Re-run this script whenever the codebase dependency graph changes.
Safe to run repeatedly; uses union semantics by default (never removes
existing entries, only adds newly discovered ones).

Usage::

    python ci/analyze_deps.py                      # update export_config.json
    python ci/analyze_deps.py --dry-run            # print diffs, no writes
    python ci/analyze_deps.py --verbose            # show per-file import traces
    python ci/analyze_deps.py --models AM MATNET   # limit model analysis
    python ci/analyze_deps.py --overwrite          # replace instead of union
    python ci/analyze_deps.py --skip-models        # skip model analysis
    python ci/analyze_deps.py --skip-policies      # skip policy analysis
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGIC_SRC = PROJECT_ROOT / "logic" / "src"
MODELS_CORE = LOGIC_SRC / "models" / "core"
SUBNETS = LOGIC_SRC / "models" / "subnets"
POLICIES = LOGIC_SRC / "policies"
HELPERS = POLICIES / "helpers"
CONFIG_FILE = PROJECT_ROOT / "ci" / "export_config.json"

_SUBNET_CATS = ("decoders", "encoders", "factories", "modules", "other")

# Category-root __init__.py files act as registries (they eagerly import all subtypes).
# Following their imports would pull in the entire subnet tree regardless of which
# model is being analyzed.  We visit them (so they're "seen") but don't follow them.
_SUBNET_REGISTRY_INITS: Set[Path] = set()  # populated after SUBNETS is defined below


def _build_registry_inits() -> None:
    # Top-level subnets/__init__.py re-exports everything — skip it too.
    _SUBNET_REGISTRY_INITS.add(SUBNETS / "__init__.py")
    for cat in (*_SUBNET_CATS, "embeddings"):
        p = SUBNETS / cat / "__init__.py"
        _SUBNET_REGISTRY_INITS.add(p)


_POLICY_CATS = (
    "constructors",
    "selectors",
    "improvement",
    "acceptance_criteria",
)

# -------------------------------------------------------------------------
# AST import extraction
# -------------------------------------------------------------------------

def _extract_imports(py_file: Path) -> List[str]:
    """Return all module-path strings imported by *py_file*, absolute only."""
    try:
        tree = ast.parse(py_file.read_text(errors="ignore"), filename=str(py_file))
    except SyntaxError:
        return []

    modules: List[str] = []
    file_pkg_parts = list(py_file.parent.relative_to(PROJECT_ROOT).parts)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if level == 0:
                if node.module:
                    modules.append(node.module)
            else:
                # Relative import: level=1 → same dir, level=2 → parent, etc.
                # Go up (level - 1) steps from the file's own directory.
                base = file_pkg_parts[: max(0, len(file_pkg_parts) - (level - 1))]
                if node.module:
                    # from .foo.bar import X  or  from ..foo import X
                    resolved = base + node.module.split(".")
                    modules.append(".".join(resolved))
                else:
                    # from . import foo, bar  →  each name is a submodule
                    for alias in node.names:
                        resolved = base + [alias.name]
                        modules.append(".".join(resolved))

    return modules


def _collect_py_files(directory: Path) -> List[Path]:
    return [
        p for p in directory.rglob("*.py")
        if "__pycache__" not in p.parts and ".venv" not in p.parts
    ]


def _module_path_to_file(module_str: str) -> Optional[Path]:
    """Resolve a dotted module path to a .py file under PROJECT_ROOT."""
    parts = module_str.replace("-", "_").split(".")
    base = PROJECT_ROOT
    for attempt in (
        base.joinpath(*parts, "__init__.py"),
        base.joinpath(*parts).with_suffix(".py"),
    ):
        if attempt.exists():
            return attempt
    return None


# -------------------------------------------------------------------------
# Transitive import tracing
# -------------------------------------------------------------------------

def _trace_imports_within(
    start_files: List[Path],
    boundary: Path,
    verbose: bool = False,
    no_follow: Optional[Set[Path]] = None,
) -> Set[Path]:
    """BFS from *start_files*, following imports that land under *boundary*.

    *no_follow*: files that are visited (recorded) but whose own imports are
    NOT followed.  Use this for "registry" __init__.py files that eagerly
    re-export everything in their package — following them would pull the
    entire subnet tree into every model's dep set regardless of actual usage.
    """
    visited: Set[Path] = set()
    queue: List[Path] = list(start_files)
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        if no_follow and current in no_follow:
            continue  # record as visited but don't follow its imports
        for mod in _extract_imports(current):
            resolved = _module_path_to_file(mod)
            if resolved is None:
                continue
            try:
                resolved.relative_to(boundary)
            except ValueError:
                continue
            if resolved not in visited:
                if verbose:
                    print(
                        f"  {current.relative_to(PROJECT_ROOT)}"
                        f" → {resolved.relative_to(PROJECT_ROOT)}"
                    )
                queue.append(resolved)
    return visited


# -------------------------------------------------------------------------
# Subnet classification helpers
# -------------------------------------------------------------------------

def _classify_subnet_file(p: Path) -> Optional[Tuple[str, str]]:
    """Return (category, name) for a file under SUBNETS, or None."""
    try:
        rel = p.relative_to(SUBNETS)
    except ValueError:
        return None
    parts = rel.parts
    if not parts:
        return None
    cat = parts[0]
    if cat not in (*_SUBNET_CATS, "embeddings"):
        return None
    if cat in ("decoders", "encoders"):
        if len(parts) < 2:
            return None
        subdir = parts[1]
        if subdir in ("common", "__pycache__") or subdir.startswith("__"):
            return None
        return (cat, subdir)
    elif cat in ("factories", "modules", "other"):
        fname = parts[-1]
        if fname.startswith("__"):
            return None
        return (cat, fname)
    elif cat == "embeddings":
        emb_rel = "/".join(parts[1:])
        if not emb_rel or parts[-1].startswith("__"):
            return None
        return ("embeddings", emb_rel)
    return None


# -------------------------------------------------------------------------
# Model analysis
# -------------------------------------------------------------------------

def analyze_model(model_dir: Path, verbose: bool = False) -> Dict[str, List[str]]:
    """Discover subnet deps (decoders/encoders/factories/modules/other) for a model."""
    model_files = _collect_py_files(model_dir)
    if not model_files:
        return {cat: [] for cat in _SUBNET_CATS}

    imported = _trace_imports_within(
        model_files, SUBNETS, verbose=verbose, no_follow=_SUBNET_REGISTRY_INITS
    )

    deps: Dict[str, Set[str]] = {cat: set() for cat in _SUBNET_CATS}
    for f in imported:
        result = _classify_subnet_file(f)
        if result and result[0] in deps:
            deps[result[0]].add(result[1])

    return {cat: sorted(deps[cat]) for cat in _SUBNET_CATS}


# -------------------------------------------------------------------------
# Embedding analysis (env-based + model-specific)
# -------------------------------------------------------------------------

def _env_name_from_embedding_rel(rel_path: str) -> Optional[str]:
    """Heuristic: stem matches a known env name."""
    stem = Path(rel_path).stem.lower()
    for env_candidate in ("ptp", "mvptp", "tcmvptp", "atsp", "tsp"):
        if stem == env_candidate:
            return env_candidate
    return None


def analyze_embeddings(
    all_env_names: List[str],
    all_model_entries: List[Tuple[str, str, Path]],
    verbose: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Return (env_files_map, model_files_map) for embeddings/ pruning."""
    emb_dir = SUBNETS / "embeddings"
    if not emb_dir.exists():
        return {}, {}

    all_emb_files = _collect_py_files(emb_dir)

    env_files: Dict[str, Set[str]] = {env: set() for env in all_env_names}
    for f in all_emb_files:
        if f.name.startswith("__"):
            continue
        rel = f.relative_to(emb_dir).as_posix()
        env_name = _env_name_from_embedding_rel(rel)
        if env_name and env_name in env_files:
            env_files[env_name].add(rel)

    all_env_file_set: Set[str] = set().union(*env_files.values())
    model_files: Dict[str, Set[str]] = {}

    for user_acr, internal, model_dir in all_model_entries:
        model_py_files = _collect_py_files(model_dir)
        imported = _trace_imports_within(
            model_py_files, SUBNETS, verbose=verbose, no_follow=_SUBNET_REGISTRY_INITS
        )
        for f in imported:
            result = _classify_subnet_file(f)
            if result is None or result[0] != "embeddings":
                continue
            rel = result[1]
            if rel not in all_env_file_set:
                model_files.setdefault(internal or user_acr.lower(), set()).add(rel)

    return (
        {env: sorted(files) for env, files in env_files.items() if files},
        {m: sorted(files) for m, files in model_files.items() if files},
    )


# -------------------------------------------------------------------------
# Policy helpers analysis
# -------------------------------------------------------------------------

def _policy_files(val: Dict, category: str) -> List[Path]:
    """Return the .py files belonging to a policy entry."""
    path_str = val.get("path", "")
    if not path_str:
        return []
    p = PROJECT_ROOT / path_str
    if p.is_dir():
        return _collect_py_files(p)
    py = p.with_suffix(".py") if p.suffix != ".py" else p
    if py.exists():
        return [py]
    return []


def analyze_policy_helpers(
    config: Dict,
    filter_cats: Optional[Set[str]] = None,
    filter_acronyms: Optional[Set[str]] = None,
    verbose: bool = False,
) -> Dict[str, List[str]]:
    """Return per-policy mapping: acronym → sorted list of helpers/ rel paths.

    Traces transitive imports: policy → helpers file → operators, etc.
    Only non-__init__.py files are included (they're always kept).
    """
    alg = config.get("algorithms", {})
    result: Dict[str, List[str]] = {}

    for cat in _POLICY_CATS:
        if filter_cats and cat not in filter_cats:
            continue
        entries = alg.get(cat, {})
        for acronym, val in entries.items():
            if acronym.startswith("_comment") or not isinstance(val, dict):
                continue
            if filter_acronyms and acronym.upper() not in filter_acronyms:
                continue

            policy_files = _policy_files(val, cat)
            if not policy_files:
                continue

            imported = _trace_imports_within(policy_files, HELPERS, verbose=verbose)

            needed: Set[str] = set()
            for f in imported:
                if f.name.startswith("__"):
                    continue
                try:
                    rel = f.relative_to(HELPERS).as_posix()
                    needed.add(rel)
                except ValueError:
                    pass

            result[acronym] = sorted(needed)

    return result


# -------------------------------------------------------------------------
# Config I/O and merge helpers
# -------------------------------------------------------------------------

def _load_config() -> Dict:
    with CONFIG_FILE.open(encoding="utf-8") as fh:
        return json.load(fh)


def _save_config(config: Dict) -> None:
    with CONFIG_FILE.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")


def _union_lists(existing: List[str], discovered: List[str]) -> List[str]:
    return sorted(set(existing) | set(discovered))


def _merge_subnet_deps(
    existing: Dict[str, List[str]],
    discovered: Dict[str, List[str]],
    overwrite: bool,
) -> Tuple[Dict[str, List[str]], bool]:
    """Merge discovered deps into existing. Returns (merged, changed)."""
    merged = {}
    changed = False
    for cat in _SUBNET_CATS:
        old = sorted(existing.get(cat, []))
        new = sorted(discovered.get(cat, []))
        if overwrite:
            merged[cat] = new
            if old != new:
                changed = True
        else:
            unioned = _union_lists(old, new)
            merged[cat] = unioned
            if unioned != old:
                changed = True
    return merged, changed


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def _process_subnet_deps(
    args: argparse.Namespace,
    models_cfg: Dict,
    filtered_model_entries: List[Tuple[str, str, Path]],
) -> List[str]:
    """Helper to analyze model subnet dependencies."""
    changed_models = []
    if args.skip_models:
        return changed_models

    print(f"[analyze_deps] Analyzing {len(filtered_model_entries)} model(s) for subnet deps …")
    for user_acr, _internal, model_dir in filtered_model_entries:
        print(f"  {user_acr} ({model_dir.name}) …", end=" ", flush=True)
        discovered = analyze_model(model_dir, verbose=args.verbose)
        print("done")

        existing_deps = models_cfg[user_acr].get("subnet_deps", {})
        merged, changed = _merge_subnet_deps(existing_deps, discovered, args.overwrite)

        if changed:
            changed_models.append(user_acr)
            if args.dry_run:
                for cat in _SUBNET_CATS:
                    old = sorted(existing_deps.get(cat, []))
                    new = merged[cat]
                    if old != new:
                        added = sorted(set(new) - set(old))
                        removed = sorted(set(old) - set(new))
                        if added:
                            print(f"    [{cat}] +{added}")
                        if removed:
                            print(f"    [{cat}] -{removed} (overwrite mode)")
            else:
                models_cfg[user_acr]["subnet_deps"] = merged
    return changed_models


def _process_embeddings(
    args: argparse.Namespace,
    prunable_types: Dict,
    all_model_entries: List[Tuple[str, str, Path]],
) -> Tuple[bool, bool]:
    """Helper to analyze model embedding dependencies."""
    if args.skip_models:
        return False, False

    print("[analyze_deps] Analyzing embedding dependencies …")
    all_env_names = list(
        prunable_types.get("embeddings", {}).get("env_files", {}).keys()
    ) or ["ptp", "mvptp"]

    env_files_map, model_files_map = analyze_embeddings(
        all_env_names, all_model_entries, verbose=args.verbose
    )

    emb_cfg = prunable_types.setdefault("embeddings", {})
    old_env_files = emb_cfg.get("env_files", {})
    old_model_files = emb_cfg.get("model_files", {})

    if args.overwrite:
        new_env_files = env_files_map
        new_model_files = model_files_map
    else:
        new_env_files = {
            env: sorted(set(old_env_files.get(env, [])) | set(files))
            for env, files in env_files_map.items()
        }
        for env, files in old_env_files.items():
            if env not in new_env_files:
                new_env_files[env] = files
        new_model_files = {
            m: sorted(set(old_model_files.get(m, [])) | set(files))
            for m, files in model_files_map.items()
        }
        for m, files in old_model_files.items():
            if m not in new_model_files:
                new_model_files[m] = files

    emb_env_changed = new_env_files != old_env_files
    emb_mod_changed = new_model_files != old_model_files

    if emb_env_changed or emb_mod_changed:
        if args.dry_run:
            if emb_env_changed:
                print(f"  [embeddings.env_files] would update {sorted(new_env_files.keys())}")
            if emb_mod_changed:
                print(f"  [embeddings.model_files] would update {sorted(new_model_files.keys())}")
        else:
            emb_cfg["env_files"] = new_env_files
            emb_cfg["model_files"] = new_model_files

    return emb_env_changed, emb_mod_changed


def _process_models(
    args: argparse.Namespace,
    config: Dict,
    prunable_types: Dict,
) -> Tuple[List[str], bool, bool]:
    """Analyze model subnet and embedding dependencies."""
    models_cfg = config.get("algorithms", {}).get("models", {})
    filter_model_upper = {m.upper() for m in args.models} if args.models else None

    # ------- Build full model entry list -------
    all_model_entries = []
    for user_acr, val in models_cfg.items():
        if user_acr.startswith("_comment") or not isinstance(val, dict):
            continue
        internal = val.get("internal_acronym", "")
        model_path_str = val.get("path", "")
        model_dir = PROJECT_ROOT / model_path_str if model_path_str else None
        if model_dir and model_dir.exists():
            all_model_entries.append((user_acr, internal, model_dir))
        elif args.verbose:
            print(f"[analyze_deps] Skipping {user_acr}: dir not found ({model_path_str})")

    filtered_model_entries = (
        [(u, i, d) for u, i, d in all_model_entries if u.upper() in filter_model_upper]
        if filter_model_upper else all_model_entries
    )

    changed_models = _process_subnet_deps(args, models_cfg, filtered_model_entries)
    emb_env_changed, emb_mod_changed = _process_embeddings(args, prunable_types, all_model_entries)

    return changed_models, emb_env_changed, emb_mod_changed


def _process_policies(args: argparse.Namespace, config: Dict) -> bool:
    """Analyze policy helpers dependencies."""
    if args.skip_policies or not HELPERS.exists():
        return False

    filter_cats = {c.lower() for c in args.policy_cats} if args.policy_cats else None
    filter_acronyms = {a.upper() for a in args.policy_acronyms} if args.policy_acronyms else None

    print("[analyze_deps] Analyzing policy helpers dependencies …")
    per_policy = analyze_policy_helpers(
        config,
        filter_cats=filter_cats,
        filter_acronyms=filter_acronyms,
        verbose=args.verbose,
    )
    print(f"  Found deps for {len(per_policy)} policies")

    existing_analysis = config.get("policies_helpers_analysis", {})
    if args.overwrite:
        new_analysis = dict(existing_analysis)
        new_analysis.update(per_policy)
    else:
        new_analysis = dict(existing_analysis)
        for acronym, files in per_policy.items():
            merged_files = sorted(set(existing_analysis.get(acronym, [])) | set(files))
            new_analysis[acronym] = merged_files

    if new_analysis != existing_analysis:
        if args.dry_run:
            added_entries = sorted(set(new_analysis) - set(existing_analysis))
            changed_entries = [
                k for k in new_analysis
                if k in existing_analysis and new_analysis[k] != existing_analysis[k]
            ]
            if added_entries:
                print(f"  Would add {len(added_entries)} new policy entries: {added_entries[:10]}")
            if changed_entries:
                print(f"  Would update {len(changed_entries)} existing policy entries")
        else:
            config["policies_helpers_analysis"] = new_analysis
        return True

    return False


def _report_and_save(
    args: argparse.Namespace,
    config: Dict,
    changed_models: List[str],
    emb_env_changed: bool,
    emb_mod_changed: bool,
    helpers_changed: bool,
) -> None:
    """Write/Report final results."""
    any_change = changed_models or emb_env_changed or emb_mod_changed or helpers_changed

    if args.dry_run:
        if changed_models:
            print(f"\n[analyze_deps] Would update subnet_deps for: {changed_models}")
        if emb_env_changed or emb_mod_changed:
            print("[analyze_deps] Would update embeddings maps")
        if helpers_changed:
            print("[analyze_deps] Would update policies_helpers_analysis")
        if not any_change:
            print("[analyze_deps] No changes needed.")
    else:
        if any_change:
            _save_config(config)
            print(f"\n[analyze_deps] Updated {CONFIG_FILE.relative_to(PROJECT_ROOT)}")
            if changed_models:
                print(f"  Models with updated subnet_deps: {changed_models}")
            if emb_env_changed or emb_mod_changed:
                print("  Embeddings maps updated")
            if helpers_changed:
                print("  policies_helpers_analysis updated")
        else:
            print("[analyze_deps] No changes — export_config.json already up to date.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ci/analyze_deps.py",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print diffs without writing.")
    parser.add_argument("--verbose", action="store_true", help="Show per-file import traces.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing deps instead of union (default: union/additive).")
    parser.add_argument("--models", nargs="*", metavar="ACRONYM",
                        help="Limit model analysis to these acronyms.")
    parser.add_argument("--skip-models", action="store_true", help="Skip model/embedding analysis.")
    parser.add_argument("--skip-policies", action="store_true", help="Skip policy helpers analysis.")
    parser.add_argument("--policy-cats", nargs="*", metavar="CAT",
                        help="Limit policy analysis to these categories "
                             f"(choices: {', '.join(_POLICY_CATS)}).")
    parser.add_argument("--policy-acronyms", nargs="*", metavar="ACRONYM",
                        help="Limit policy analysis to these acronyms.")
    args = parser.parse_args(argv)

    _build_registry_inits()

    if not CONFIG_FILE.exists():
        print(f"ERROR: {CONFIG_FILE} not found", file=sys.stderr)
        sys.exit(1)

    config = _load_config()
    subnet_pruning = config.setdefault("subnet_pruning", {})
    prunable_types = subnet_pruning.setdefault("prunable_types", {})

    changed_models, emb_env_changed, emb_mod_changed = _process_models(
        args, config, prunable_types
    )

    helpers_changed = _process_policies(args, config)

    _report_and_save(
        args,
        config,
        changed_models,
        emb_env_changed,
        emb_mod_changed,
        helpers_changed,
    )


if __name__ == "__main__":
    main()
