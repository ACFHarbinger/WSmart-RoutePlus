"""Build the WSmart-Route release package from ``logic/release.yaml``.

The manifest is the single source of truth for the released package (owner
ruling Q-10a): registry keys per stage, released problems, and the
infrastructure seeds whose import closure ships. Everything else under ``logic/``
stays in ``main`` and is pruned from the package (Q-8/Q-8b).

Usage::

    # CI gate: every manifest key is registered; seeds and configs resolve.
    python logic/package/build_release.py --check

    # Build the release tree into <dir> and verify its import closure.
    python logic/package/build_release.py --build <dir>

    # Torch-free classical smoke test against a built tree (fresh
    # interpreter; CI runs it in a venv without torch installed).
    python logic/package/build_release.py --smoke <dir>

The build works on a staged copy; the working tree is never modified. Unreleased
code is not migrated or changed here — it simply does not ship (Q-8b migration
belongs to the config and simulator work packages).

Retired with this tool (Q-10a): ``ci/packager.py``, ``ci/export_config.json``,
``logic/package/prune_codebase.py`` and the ``remove_*.py`` scripts, and the
``ci/export_profiles/`` patch flow. The manifest uses registry keys, not the old
BB/BC/BPC acronyms; the Studio app keeps its own display-name map.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

if not __package__:
    # Script-style invocation: python logic/package/build_release.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from logic.package.release_closure import (
    LOGIC,
    REPO,
    STAGES,
    compute_closure,
    file_to_module,
    module_to_file,
    parse_registry_manifest,
    release_tests,
)
from logic.package.release_files import (
    prune_policy_defaults,
    released_stage_configs,
    write_docs_index,
    write_release_pyproject,
)
from logic.package.release_staging import (
    drop_pruned_init_imports,
    filter_lazy_map,
    filter_manifest_file,
    import_sweep,
    prune_unreferenced,
    unguarded_missing_imports,
    unresolved_names,
    validate_release_tests,
    verify_staged,
    write_release_conftest,
    write_single_manifest,
)

# logic/ subtrees that ship in the release. Anything else (benchmark, cloud,
# docs, gen, package, validation) is development tooling and stays out (Q-11b).
STAGED_TOPS = ("src", "configs", "controllers")


# --------------------------------------------------------------------------- #
# Manifest loading                                                            #
# --------------------------------------------------------------------------- #


DEFAULT_MANIFEST = LOGIC / "release.yaml"


def load_manifest(path: Optional[Path] = None) -> dict:
    """Parse a release manifest (default ``logic/release.yaml``) and return its plain-dict form.

    ``_path`` records the file, so the build stages it and resolves its overlay and patches.
    """
    import yaml

    path = Path(path or DEFAULT_MANIFEST).resolve()
    with path.open() as fh:
        manifest = yaml.safe_load(fh)
    manifest["_path"] = str(path)
    for section in ("version", "problems", "stages", "infrastructure"):
        if section not in manifest:
            raise SystemExit(f"release.yaml: missing required section {section!r}")
    manifest.setdefault("excluded_packages", [])
    manifest.setdefault("tasks", None)  # None: every task config ships
    return manifest


def prune_task_configs(tasks_dir: Path, released: Optional[List[str]]) -> List[str]:
    """Delete the task YAMLs of unreleased tasks from a staged ``configs/tasks``; return their names."""
    if released is None:
        return []
    removed = []
    for task_yaml in sorted(tasks_dir.glob("*.yaml")):
        if task_yaml.stem not in released:
            task_yaml.unlink()
            removed.append(task_yaml.stem)
    return removed


# --------------------------------------------------------------------------- #
# Module/file mapping (works for the repo tree or a staged build tree)        #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Import analysis                                                             #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Closure computation                                                         #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Staging transforms                                                           #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Check / build / smoke                                                        #
# --------------------------------------------------------------------------- #


def run_check(manifest: dict) -> int:  # noqa: C901
    problems: List[str] = []
    stage_entries = {stage: parse_registry_manifest(path) for stage, path in STAGES.items()}
    for stage, keys in manifest["stages"].items():
        for key in keys:
            if key not in stage_entries[stage]:
                problems.append(f"stage {stage}: key {key!r} is not registered in _manifest.py")
    for problem_name in manifest["problems"]:
        if not (LOGIC / "src" / "envs" / f"{problem_name}.py").is_file():
            problems.append(f"problem {problem_name!r}: no envs/{problem_name}.py")
    for seed in manifest["infrastructure"]:
        if module_to_file(seed, REPO) is None:
            problems.append(f"infrastructure seed {seed!r} does not resolve to a file")
    for excluded in manifest["excluded_packages"]:
        if module_to_file(excluded, REPO) is None:
            problems.append(f"excluded package {excluded!r} does not resolve to a file")
    for task in manifest["tasks"] or ():
        if not (LOGIC / "configs" / "tasks" / f"{task}.yaml").is_file():
            problems.append(f"task {task!r}: no configs/tasks/{task}.yaml")
    routing = set(manifest["stages"].get("routing", ()))
    packing = set(manifest["stages"].get("packing", ()))
    for key in routing:
        if not any((LOGIC / "configs" / "policies").rglob(f"policy_{key}.yaml")):
            problems.append(f"routing key {key!r}: no configs/policies/**/policy_{key}.yaml")
    for key in packing:
        if not any((LOGIC / "configs" / "policies" / "other").rglob(f"pk_{key}.yaml")):
            problems.append(f"packing key {key!r}: no configs/policies/other/**/pk_{key}.yaml")
    for stage, keys in manifest["stages"].items():
        for key in keys:
            target = stage_entries[stage].get(key, "")
            if target and module_to_file(target.split(":")[0], REPO) is None:
                problems.append(f"stage {stage} key {key!r}: target module missing ({target})")
    if problems:
        print("release manifest check FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    total = sum(len(keys) for keys in manifest["stages"].values())
    print(
        f"release manifest check OK: {total} stage keys, {len(manifest['problems'])} problems, "
        f"{len(manifest['infrastructure'])} infrastructure seeds"
    )
    return 0


def staged_module_names(out: Path) -> set:
    """Modules of the staged tree.

    The bin-analysis files are staged under their importable name, ``wsmart_bin_analysis``.
    """
    return {file_to_module(p, out) for p in (out / "logic").rglob("*.py")}


def make_test_venv(out: Path, venv: Path, python_version: str = "3.12") -> str:
    """Install the staged release (solvers, neural and test extras) into ``venv``; return its interpreter.

    The release tests then run against the package's own pinned dependencies, as a user's
    setup script installs them, not against main's environment.
    """
    venv = venv.resolve()
    if not (venv / "bin" / "python").exists():
        subprocess.run(["uv", "venv", "-q", "--python", python_version, str(venv)], check=True)
    env = dict(__import__("os").environ, VIRTUAL_ENV=str(venv))
    subprocess.run(
        ["uv", "pip", "install", "-q", "--reinstall-package", "wsmart-route", f"{out}[solvers,neural,test]"],
        check=True,
        env=env,
    )
    for leftover in [out / "build", *out.glob("*.egg-info")]:
        shutil.rmtree(leftover, ignore_errors=True)
    return str(venv / "bin" / "python")


def run_build(  # noqa: C901
    manifest: dict,
    out: Path,
    test_python: Optional[str] = None,
    test_data: Optional[Path] = None,
    test_venv: Optional[Path] = None,
) -> int:
    out = out.resolve()
    if out in (REPO, LOGIC) or out in REPO.parents or LOGIC in out.parents:
        raise ValueError("Release destination must not contain the repository or lie inside logic/")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Release destination must be absent or empty; existing files are never deleted")
    kept, notes = compute_closure(manifest)

    # --- stage the copy -------------------------------------------------- #
    (out / "logic").mkdir(parents=True)
    for top in STAGED_TOPS:
        source = LOGIC / top
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if path.is_dir() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(LOGIC)
            if path.suffix == ".py" and path not in kept:
                continue
            dest = out / "logic" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    shutil.copy2(manifest.get("_path", DEFAULT_MANIFEST), out / "logic" / "release.yaml")
    # logic must be a regular package in the staging tree; without __init__.py
    # it degrades to a namespace package and loses to any other installed copy.
    shutil.copy2(LOGIC / "__init__.py", out / "logic" / "__init__.py")

    # --- the bin-analysis package (RQ-grok-7 a): runtime modules only ------ #
    bin_analysis = REPO / "logic" / "submodules" / "WSmart-Bin-Analysis"
    if not (bin_analysis / "__init__.py").is_file():
        raise SystemExit(
            f"{bin_analysis} is empty: run git submodule update --init logic/submodules/WSmart-Bin-Analysis"
        )
    for path in [
        bin_analysis / "__init__.py",
        *sorted((bin_analysis / "export").rglob("*.py")),
        *sorted((bin_analysis / "python").rglob("*.py")),
    ]:
        if "__pycache__" in path.parts or "notebooks" in path.parts or "docs" in path.parts:
            continue
        if path.stem.endswith("_experiment"):  # research scripts (statsmodels); nothing imports them
            continue
        # A real directory under the name the code imports: a symlink in the package
        # becomes a plain file in a zip or a Windows checkout, and PyInstaller's
        # collect_submodules("logic") does not see a hyphenated directory.
        dest = out / "logic" / "submodules" / "wsmart_bin_analysis" / path.relative_to(bin_analysis)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    # --- registry manifests: released keys only -------------------------- #
    for stage, stage_path in STAGES.items():
        rel = stage_path.relative_to(LOGIC)
        staged_manifest = out / "logic" / rel
        if staged_manifest.is_file():
            notes += filter_manifest_file(staged_manifest, set(manifest["stages"].get(stage, ())))
    if manifest.get("single_manifest"):
        stage_files = {stage: out / "logic" / path.relative_to(LOGIC) for stage, path in STAGES.items()}
        notes += write_single_manifest(out, stage_files)

    # --- package configs: released YAML only ----------------------------- #
    routing = set(manifest["stages"].get("routing", ()))
    for yaml_path in sorted((out / "logic" / "configs" / "policies").rglob("policy_*.yaml")):
        if yaml_path.stem.removeprefix("policy_") not in routing:
            yaml_path.unlink()
    other_dir = out / "logic" / "configs" / "policies" / "other"
    if other_dir.is_dir():
        keep = released_stage_configs(manifest, out / "logic" / "configs" / "policies")
        for yaml_path in sorted(other_dir.rglob("*.yaml")):
            if yaml_path.name not in keep:
                yaml_path.unlink()
    # The policy configs are grouped by category; drop the groups the release empties.
    for folder in sorted((out / "logic" / "configs" / "policies").rglob("*"), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()

    # --- tests: release CI set only (P-06) ------------------------------- #
    release, pruned = release_tests(manifest, kept)
    test_out = out / "logic" / "test"
    for path in release:
        rel = path.relative_to(LOGIC / "test")
        dest = test_out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    # --- lazy maps and eager init imports -------------------------------- #
    staged_modules = staged_module_names(out)
    # Deepest first: an init's re-exports are checked against the already pruned inits below it.
    for staged in sorted((out / "logic").rglob("__init__.py"), key=lambda p: (-len(p.parts), str(p))):
        notes += filter_lazy_map(staged, staged_modules, out)
        notes += drop_pruned_init_imports(staged, staged_modules, out)

    # --- dead code: definitions nothing references (profiles that ask for it) #
    if manifest.get("prune_unreferenced"):
        notes += prune_unreferenced(out, set(manifest.get("keep_names", [])))

    # --- excluded packages: drop the data files their pruned code read ----- #
    for excluded in manifest["excluded_packages"]:
        folder = out / Path(*excluded.split("."))
        if folder.is_dir() and not any(folder.rglob("*.py")):
            shutil.rmtree(folder)
            notes.append(f"{excluded}: removed its data files (no code ships)")

    # --- task configs: only the released tasks ship ----------------------- #
    removed_tasks = prune_task_configs(out / "logic" / "configs" / "tasks", manifest["tasks"])
    if removed_tasks:
        notes.append(f"configs/tasks: dropped the unreleased task configs {', '.join(removed_tasks)}")

    # --- env configs: only the profile's problems (profiles that ask for it) - #
    if manifest.get("only_problem_env_configs"):
        for env_yaml in sorted((out / "logic" / "configs" / "envs").glob("*.yaml")):
            if env_yaml.stem not in manifest["problems"]:
                env_yaml.unlink()
                notes.append(f"configs/envs: dropped {env_yaml.name} (not a {manifest.get('name', 'release')} problem)")

    # --- task defaults: drop the policy groups that do not ship ------------ #
    for task_yaml in sorted((out / "logic" / "configs" / "tasks").glob("*.yaml")):
        dropped = prune_policy_defaults(task_yaml, out / "logic" / "configs" / "policies")
        if dropped:
            notes.append(f"{task_yaml.name}: dropped {dropped} unreleased policy defaults")

    # --- entry point: `python -m <release dir>` / `python __main__.py <task>` #
    # The Hydra dispatcher it imports is an infrastructure seed; main.py's
    # non-Hydra jobs (test_suite, benchmark, cloud sync) are not released.
    shutil.copy2(REPO / "__main__.py", out / "__main__.py")

    # --- release pyproject + docs index ---------------------------------- #
    write_release_pyproject(manifest, out)
    write_docs_index(manifest, out)

    # --- release tests: only those that pass in the staged tree ----------- #
    dropped_plugins = write_release_conftest(LOGIC / "test" / "conftest.py", out)
    if dropped_plugins:
        notes.append(f"conftest.py: dropped the pruned fixture plugins {', '.join(dropped_plugins)}")
    if test_venv is not None:
        test_python = make_test_venv(out, test_venv)
    import_failures = import_sweep(out, test_python)
    if import_failures:
        raise SystemExit("release modules fail to import:\n  " + "\n  ".join(import_failures))
    staged_tests = [test_out / p.relative_to(LOGIC / "test") for p in release]
    release, failures = validate_release_tests(out, staged_tests, python=test_python, data_dir=test_data)
    for path in set(staged_tests) - set(release):
        path.unlink()
    (test_out / "release_tests.txt").write_text("\n".join(str(p.relative_to(test_out)) for p in release) + "\n")
    if failures:
        (test_out / "release_tests.excluded.txt").write_text(
            "# Not shipped: these tests fail in the release tree (they need excluded code, the repository's\n"
            "# tools/ or main.py, or behaviour the release leaves out). First failure per file:\n"
            + "\n".join(failures)
            + "\n"
        )
        notes.append(f"{len(failures)} test files dropped: they fail in the release tree")

    # --- verification ----------------------------------------------------- #
    staged_modules = staged_module_names(out)
    errors = verify_staged(out, staged_modules)
    notes += [p for p in unresolved_names(out, staged_modules) if p.startswith("deferred")]
    notes += unguarded_missing_imports(out, staged_modules)
    staged_py = list((out / "logic").rglob("*.py"))
    lines = sum(len(p.read_text().splitlines()) for p in staged_py)
    print(
        f"built {out}: {len(staged_py)} python files, {lines} lines, "
        f"{len(release)} release test files (pruned {len(pruned)})"
    )
    for note in notes:
        print(f"  note: {note}")
    if errors:
        print("staged closure verification FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("staged closure verification OK: no module-level import escapes the manifest")
    return 0


SMOKE_SCRIPT = r"""
import sys
import importlib.abc

class NoNeural(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "tensordict", "torchrl", "lightning", "pytorch_lightning", "wandb"}:
            raise ImportError("neural extra unavailable during classical smoke: " + fullname)

sys.meta_path.insert(0, NoNeural())
sys.path.insert(0, sys.argv[1])
assert "torch" not in sys.modules

from logic.src.policies.packing.base.registry import PackingRegistry  # noqa: F401
from logic.src.policies.route_improvement.base.registry import RouteImproverRegistry  # noqa: F401
from logic.src.policies.acceptance_criteria.base.registry import AcceptanceCriterionRegistry  # noqa: F401
from logic.src.utils.registry import ACCEPTANCE, IMPROVEMENT, PACKING

assert PACKING.get("last_minute") is not None
assert PACKING.get("service_level") is not None
assert PACKING.get("lookahead") is not None
assert PACKING.get("regular") is not None
assert IMPROVEMENT.get("classical_local_search") is not None
assert IMPROVEMENT.get("fast_tsp") is not None
assert ACCEPTANCE.get("bmc") is not None
assert ACCEPTANCE.get("oi") is not None

# Unreleased keys must not resolve (Q-8).
assert PACKING.get("revenue_threshold") is None, "unreleased packing key resolves"
assert ACCEPTANCE.get("lahc") is None, "unreleased acceptance key resolves"
assert IMPROVEMENT.get("lkh") is None, "unreleased improver key resolves"

PACKING.create("last_minute")
PACKING.create("regular")
ACCEPTANCE.create("bmc", initial_temp=100.0, alpha=0.995)
ACCEPTANCE.create("oi")
IMPROVEMENT.create("classical_local_search")
IMPROVEMENT.create("fast_tsp")

assert "torch" not in sys.modules, "torch was imported on the classical path"
print("smoke OK: classical components resolve without torch")

# Resolve every classical routing adapter and the simulator with neural imports blocked.
# NA remains declared, and is resolved only when the neural extra is requested.
from logic.src.policies.routing.base.registry import RoutingAlgorithmRegistry  # noqa: F401
from logic.src.utils.registry import ROUTING

for key in ("swc_tcf", "bpc", "alns", "sans", "hgs", "psoma", "aco_hh", "pg_clns", "cvrp"):
    assert ROUTING.get(key) is not None, f"released routing key {key} does not resolve"
assert "na" in ROUTING.names(), "released neural policy is not declared"
assert ROUTING.get("hgs_alns") is None, "unreleased routing key resolves"
import logic.src.pipeline.simulations.manager  # noqa: F401
from logic.submodules.wsmart_bin_analysis import GridBase  # noqa: F401

assert "torch" not in sys.modules
print("smoke OK: classical routing keys and simulator import without the neural extra")

# The cloud runner without the [cloud] extra: provider modules import (their SDKs are lazy),
# the CLI parses, and a local job runs end to end against a scripted backend.
class NoCloudSdk(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"boto3", "botocore", "azure", "supabase", "keyring"} or fullname.startswith("google.cloud"):
            raise ImportError("cloud extra unavailable during the smoke: " + fullname)
        return None

sys.meta_path.insert(0, NoCloudSdk())
import importlib as _il
from logic.controllers.cloud.providers import BACKENDS
for target in sorted(set(BACKENDS.values())):
    module, _, cls = target.partition(":")
    assert hasattr(_il.import_module(module), cls), target
for provider in ("gcp", "aws", "azure"):
    _il.import_module(f"logic.controllers.cloud.providers.{provider}.storage")
    assert _il.import_module(f"logic.controllers.cloud.providers.{provider}.pricing").price_per_hour is not None
import logic.controllers.cloud.__main__  # noqa: F401
from logic.controllers.cloud.model import CellSpec, JobSpec, TaskState
from logic.controllers.cloud.local import SQLiteControlPlane
from logic.controllers.cloud.service import CloudService
from logic.controllers.cloud.testing import FakeClock, MemoryObjectStore, ScriptedBackend
from logic.controllers.cloud.model import ComputeKind, Provider
import tempfile
service = CloudService(
    control_plane=SQLiteControlPlane(":memory:"),
    store=MemoryObjectStore(),
    backends={(Provider.LOCAL, ComputeKind.PROCESS): ScriptedBackend()},
    workroot=tempfile.mkdtemp(),
    clock=FakeClock(),
)
job = service.submit(JobSpec(name="smoke", cells=[CellSpec(cell_id="c", argv=["test_sim"])]))
for _ in range(5):
    service.tick()
assert [t.state for t in service.cp.list_tasks(job.job_id)] == [TaskState.SUCCEEDED]
from logic.controllers.cloud.prediction import default_predictor
assert default_predictor().predict([CellSpec(cell_id="30d/riomaior/100/emp/la_cls/lookahead/alns", argv=["test_sim"])])[0].p50
assert not any(m.startswith(("boto3", "google.cloud", "azure")) for m in sys.modules)
print("smoke OK: cloud runner imports and runs a local job without the cloud extra")
"""


def run_smoke(out: Path) -> int:
    out = out.resolve()
    if not (out / "logic").is_dir():
        raise SystemExit(f"{out} is not a built release tree (run --build first)")
    result = subprocess.run(
        [sys.executable, "-c", SMOKE_SCRIPT, str(out)],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="")
        print("smoke FAILED")
        return 1
    return 0


def prepare_source(manifest: dict, source: Path, tracked: Optional[List[str]] = None) -> Path:
    """Copy the tracked source to ``source`` and apply the manifest's overlay, deletions and patches.

    The overlay directory holds files added to or replacing the source at the same relative
    path; ``delete`` lists source paths to remove; patches are ``git format-patch`` files (applied with ``git apply``
    in name order) that fail the build when they no longer apply. Returns the prepared root.
    """
    base = Path(manifest["_path"]).parent
    if source.exists():
        shutil.rmtree(source)
    source.mkdir(parents=True)
    if tracked is None:  # every tracked file, submodules included
        listing = subprocess.run(
            ["git", "ls-files", "-z", "--recurse-submodules"], cwd=REPO, capture_output=True, check=True
        ).stdout.decode()
        tracked = [rel for rel in listing.split("\0") if rel]
    for rel in tracked:
        src = REPO / rel
        if src.is_file() and not src.is_symlink():
            dest = source / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    overlay = base / manifest["overlay"] if manifest.get("overlay") else None
    if overlay is not None and overlay.is_dir():
        for src in sorted(overlay.rglob("*")):
            if src.is_file() and "__pycache__" not in src.parts:
                dest = source / src.relative_to(overlay)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
    for deleted in manifest.get("delete", []):
        target = source / deleted
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    patches = base / manifest["patches"] if manifest.get("patches") else None
    if patches is not None and patches.is_dir():
        for patch in sorted(patches.glob("*.patch")):
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch)], cwd=source, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise SystemExit(f"patch {patch.name} does not apply:\n{result.stdout}{result.stderr}")
    return source


def run_violations(manifest: dict) -> int:
    """List every kept module-level import of an excluded package (what a profile still has to patch)."""
    manifest["_violations"] = []
    _, notes = compute_closure(manifest)
    # A module reached through a package's public surface whose chain hard-imports an excluded
    # package is dropped, not reported; for a profile that is a feature silently lost.
    dropped = [note for note in notes if note.startswith("surface module")]
    found = sorted(set(manifest["_violations"])) + sorted(set(dropped))
    for violation in found:
        print(violation)
    print(f"{len(found)} module-level imports of excluded packages or dropped surface modules")
    return 1 if found else 0


def run_why(manifest: dict, target: str) -> int:
    """Print, for each kept module under ``target``, the chain of modules that pulls it in."""
    why: Dict[str, str] = {}
    compute_closure(manifest, why)
    hits = sorted(module for module in why if module == target or module.startswith(target + "."))
    if not hits:
        print(f"{target}: not in the release")
        return 1
    for module in hits:
        chain, current = [module], why[module]
        while current and len(chain) < 40:
            chain.append(current)
            current = why.get(current.split(" ")[0], "")
        print(
            "  <- ".join(chain[:1])
            + "".join(f"\n    <- {link}" for link in chain[1:])
            + ("\n    <- (seed)" if not current else "")
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="validate the manifest against the tree")
    parser.add_argument("--build", metavar="DIR", help="build the release tree into DIR")
    parser.add_argument(
        "--test-python",
        metavar="PY",
        help="interpreter of the release environment that runs the release tests (default: this one)",
    )
    parser.add_argument(
        "--test-venv",
        metavar="DIR",
        help="create or update a venv in DIR with the staged release and run the release tests in it",
    )
    parser.add_argument(
        "--test-data", metavar="DIR", help="the package data folder, linked in as data/ while the tests run"
    )
    parser.add_argument("--smoke", metavar="DIR", help="torch-free smoke test against a built tree")
    parser.add_argument(
        "--why", metavar="MODULE", help="print the import chain that ships MODULE (or a package prefix)"
    )
    parser.add_argument("--manifest", metavar="YAML", help="release manifest (default logic/release.yaml)")
    parser.add_argument(
        "--source-dir",
        metavar="DIR",
        help="where a manifest with an overlay or patches prepares its source (default ~/.cache/wsr-build/<name>)",
    )
    parser.add_argument("--violations", action="store_true", help="list kept module-level imports of excluded packages")
    args = parser.parse_args()
    manifest = load_manifest(Path(args.manifest) if args.manifest else None)
    changes_source = manifest.get("overlay") or manifest.get("patches") or manifest.get("delete")
    if changes_source and not os.environ.get("WSR_BUILD_SOURCE"):
        name = manifest.get("name", Path(manifest["_path"]).parent.name)
        source = Path(args.source_dir or Path.home() / ".cache" / "wsr-build" / str(name)).resolve()
        prepare_source(manifest, source)
        print(f"prepared source: {source}")
        env = dict(os.environ, WSR_BUILD_SOURCE=str(source))
        return subprocess.run([sys.executable, *sys.argv], env=env).returncode
    if args.violations:
        return run_violations(manifest)
    if args.why:
        return run_why(manifest, args.why)
    if args.check:
        return run_check(manifest)
    if args.build:
        return run_build(
            manifest,
            Path(args.build),
            args.test_python,
            Path(args.test_data) if args.test_data else None,
            Path(args.test_venv) if args.test_venv else None,
        )
    if args.smoke:
        return run_smoke(Path(args.smoke))
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "load_manifest",
    "prune_task_configs",
    "filter_manifest_file",
    "filter_lazy_map",
    "drop_pruned_init_imports",
    "validate_release_tests",
    "run_check",
    "run_build",
    "verify_staged",
    "write_release_pyproject",
    "released_stage_configs",
    "prune_policy_defaults",
    "write_docs_index",
    "run_smoke",
    "main",
    "REPO",
    "LOGIC",
    "STAGES",
    "parse_registry_manifest",
    "module_to_file",
    "file_to_module",
    "compute_closure",
    "release_tests",
]
