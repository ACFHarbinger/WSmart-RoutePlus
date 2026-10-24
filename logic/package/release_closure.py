"""Import closure of the release: which ``logic/`` modules and tests the manifest's seeds reach.

Attributes:
    REPO: Repository root.
    STAGES: Stage name -> registry manifest file.
    INDEXED_INITS: Package inits whose eager re-exports are indexed, not followed.
    compute_closure: Every module reachable from the released stage keys and infrastructure seeds.
    release_tests: The curated tests whose imports stay inside the closure.

Example:
    >>> kept, missing = compute_closure(load_manifest())
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[2]


LOGIC = REPO / "logic"


STAGES: Dict[str, Path] = {
    "routing": LOGIC / "src" / "policies" / "routing" / "_manifest.py",
    "packing": LOGIC / "src" / "policies" / "packing" / "_manifest.py",
    "route_improvement": LOGIC / "src" / "policies" / "route_improvement" / "_manifest.py",
    "acceptance": LOGIC / "src" / "policies" / "acceptance_criteria" / "_manifest.py",
    "packing_and_routing": LOGIC / "src" / "policies" / "packing_and_routing" / "_manifest.py",
}


ENTRY_RE = re.compile(r"""^\s*['"]([A-Za-z0-9_-]+)['"]:\s*['"](logic\.[A-Za-z0-9_.:]+)['"]\s*,?\s*$""")


LAZY_LINE_RE = re.compile(r'^(\s*)"([A-Za-z_][A-Za-z0-9_]*)":\s*"(logic\.[A-Za-z0-9_.:]+)"\s*,?\s*$', re.MULTILINE)


LOGIC_MODULE_RE = re.compile(r"\blogic(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")


# Package inits whose module-level re-exports are indexed like a lazy map instead of
# followed. ``logic.src.configs.policies`` imports the config dataclass of every policy
# family; following it would ship the configs of every unreleased policy. A config ships
# when a kept module imports it, by module path or by name from the package.
INDEXED_INITS = ("logic.src.configs.policies",)


def lazy_targets(path: Path) -> Dict[str, str]:
    """Read literal lazy re-export maps independently of quote style or line wrapping."""
    for node in ast.parse(path.read_text()).body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id == "_LAZY" for target in targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                return {
                    key: item.split(":")[0]
                    for key, item in value.items()
                    if isinstance(key, str) and isinstance(item, str)
                }
    return {}


def eager_targets(path: Path, root: Path) -> Dict[str, str]:
    """Map each name a package init imports at module level from ``logic.*`` to its module."""
    targets: Dict[str, str] = {}
    pkg = _package_of(path, root)
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.ImportFrom) or node.module is None and node.level == 0:
            continue
        if node.level == 0:
            module = node.module
        else:
            up = pkg[: len(pkg) - (node.level - 1)]
            module = ".".join(up + ([node.module] if node.module else []))
        if not module.startswith("logic"):
            continue
        for alias in node.names:
            targets[alias.asname or alias.name] = module
    return targets


def package_index(path: Path, module: str, root: Path) -> Dict[str, str]:
    """Name -> target module served by a package init: its lazy map, plus eager re-exports if indexed."""
    index = lazy_targets(path)
    if module in INDEXED_INITS:
        index.update(eager_targets(path, root))
    return index


def parse_registry_manifest(path: Path) -> Dict[str, str]:
    """Return {registry key: 'module:Class'} from a generated ``_manifest.py``."""
    entries: Dict[str, str] = {}
    in_entries = False
    for line in path.read_text().splitlines():
        if line.startswith("ENTRIES"):
            in_entries = True
            continue
        if in_entries:
            if line.startswith("}"):
                in_entries = False
                continue
            match = ENTRY_RE.match(line)
            if match:
                entries[match.group(1)] = match.group(2)
    return entries


# Import names whose files sit under another folder: the bin-analysis submodule's folder name has
# a hyphen, and logic/__init__.py maps the import name onto it.
MODULE_ALIASES = {"logic.submodules.wsmart_bin_analysis": ("logic", "submodules", "WSmart-Bin-Analysis")}


def module_to_file(module: str, root: Path) -> Optional[Path]:
    """Map a dotted module under ``<root>/logic`` to its file, or None."""
    parts = module.split(".")
    for alias, folder in MODULE_ALIASES.items():
        alias_parts = alias.split(".")
        if parts[: len(alias_parts)] == alias_parts and not root.joinpath(*alias_parts).exists():
            parts = [*folder, *parts[len(alias_parts) :]]
            break
    candidate = root.joinpath(*parts)
    if candidate.with_suffix(".py").is_file():
        return candidate.with_suffix(".py")
    if (candidate / "__init__.py").is_file():
        return candidate / "__init__.py"
    return None


def file_to_module(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def trimmed(module: str, root: Path) -> str:
    """Longest prefix of ``module`` that maps to a file under ``root``."""
    parts = module.split(".")
    while parts:
        candidate = ".".join(parts)
        if module_to_file(candidate, root) is not None:
            return candidate
        parts = parts[:-1]
    return ""


@dataclass
class Imports:
    module_level: Set[str] = field(default_factory=set)
    deferred: Set[str] = field(default_factory=set)


def _package_of(path: Path, root: Path) -> List[str]:
    """Dotted package path containing ``path`` (the module's own name stripped)."""
    parts = list(path.relative_to(root).with_suffix("").parts)
    parts = parts[:-1]  # works for both module files and __init__.py
    return parts


def imports_of(path: Path, root: Path) -> Imports:
    """Collect logic.* imports of a file, split by module level vs deferred."""
    result = Imports()
    tree = ast.parse(path.read_text())
    pkg = _package_of(path, root)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = (node.module or "").split(".")
                names = [".".join(base + [alias.name]) for alias in node.names]
                if node.module:
                    names.append(node.module)
            else:
                up = pkg[: len(pkg) - (node.level - 1)]
                base = up + ([node.module] if node.module else [])
                names = [".".join(base + [alias.name]) for alias in node.names]
                names.append(".".join(base))
        else:
            continue
        for name in names:
            if not name.startswith("logic"):
                continue
            name = name.rstrip(".")
            if name and _is_deferred(tree, node):
                result.deferred.add(name)
            else:
                result.module_level.add(name)
    return result


def _is_deferred(tree: ast.Module, target: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if inner is target:
                    return True
    return False


def compute_closure(manifest: dict) -> Tuple[Set[Path], List[str]]:  # noqa: C901
    """Keep the module-level import closure of the seeds minus exclusions."""
    excluded = tuple(manifest["excluded_packages"])
    stage_entries = {stage: parse_registry_manifest(path) for stage, path in STAGES.items()}
    seeds: Set[str] = set(manifest["infrastructure"])
    for stage, keys in manifest["stages"].items():
        for key in keys:
            target = stage_entries[stage].get(key)
            if target is None:
                raise SystemExit(f"release.yaml: stage {stage!r} key {key!r} is not registered")
            seeds.add(target.split(":")[0])

    kept_modules: Set[str] = set()
    notes: List[str] = []

    def enqueue(module: str) -> None:
        """Add a module and all its ancestor packages to the work queue."""
        parts = module.split(".")
        for depth in range(len(parts), 0, -1):
            candidate = ".".join(parts[:depth])
            if module_to_file(candidate, REPO) is None:
                continue
            if candidate not in kept_modules and not candidate.startswith(excluded):
                queue.append(candidate)
            if candidate in kept_modules:
                continue

    # name -> target module, harvested from _LAZY maps in package inits:
    # resolves "from <pkg> import <Name>" when the name lives in a submodule.
    lazy_index: Dict[str, Dict[str, str]] = {}
    # Modules discovered through a package's public surface (lazy map or
    # deferred re-export). If such a module's chain hard-imports an excluded
    # package, the module is unreleasable and is dropped from the closure:
    # the code path is simply unavailable in the release (Q-6/Q-8).
    surface: Set[str] = set()

    queue: List[str] = list(seeds)
    while queue:
        module = queue.pop()
        root = trimmed(module, REPO)
        if not root or root in kept_modules:
            continue
        if root.startswith(excluded):
            continue
        kept_modules.add(root)
        # The module's ancestor packages ship too (skeleton), and their eager
        # imports are part of the closure.
        enqueue(root)
        path = module_to_file(root, REPO)
        if path is None:  # pragma: no cover - trimmed() guarantees a file
            continue
        imports = imports_of(path, REPO)
        # Package re-export maps and deferred imports of kept __init__ files
        # are part of the package's public surface. Deferred imports are the
        # runtime opt-in pattern: an excluded target is skipped with a note
        # (the code path is simply unavailable in the release).
        deferred: Set[str] = set(imports.deferred)
        pending: List[str] = sorted(imports.module_level | deferred)
        if path.name == "__init__.py":
            # A lazy map is indexed, not followed: a target ships only when a kept
            # module imports its name (below). Following every entry would ship
            # every architecture and algorithm behind the facades (Q-12).
            lazy_index[root] = package_index(path, root, REPO)
            surface.update(trimmed(dep, REPO) for dep in imports.deferred)
            if root in INDEXED_INITS:
                pending = sorted(deferred)
        for dep in pending:
            root_dep = trimmed(dep, REPO)
            if not root_dep:
                notes.append(f"unresolved module-level import in {root}: {dep}")
                continue
            # "from <pkg> import <Name>" where Name is served by the
            # package's lazy map: enqueue the real target module.
            if module_to_file(dep, REPO) is None and dep != root_dep:
                name = dep[len(root_dep) + 1 :]
                if "." not in name:
                    if root_dep not in lazy_index:
                        package_path = module_to_file(root_dep, REPO)
                        lazy_index[root_dep] = (
                            package_index(package_path, root_dep, REPO) if package_path is not None else {}
                        )
                    target = lazy_index.get(root_dep, {}).get(name)
                    if target and target not in kept_modules:
                        surface.add(trimmed(target, REPO))
                        queue.append(target)
            if root_dep.startswith(excluded):
                if path.name != "__init__.py" and dep not in deferred:
                    if root in surface:
                        kept_modules.discard(root)
                        notes.append(
                            f"surface module {root} dropped: its chain module-level "
                            f"imports excluded {root_dep} (path unavailable in the release)"
                        )
                        break
                    raise SystemExit(
                        f"kept file {path} imports excluded package {root_dep!r} at module "
                        "level; move the import into a function, or extend the manifest"
                    )
                notes.append(f"excluded {root_dep} stays out of the release ({path.name})")
                continue
            if root_dep not in kept_modules:
                queue.append(root_dep)

    files: Set[Path] = set()
    for module in kept_modules:
        path = module_to_file(module, REPO)
        if path is not None:
            files.add(path)
    # The _manifest.py files always ship (filtered to released keys).
    for stage_path in STAGES.values():
        files.add(stage_path)
    return files, notes


FROM_IMPORT_RE = re.compile(r"^\s*from (logic(?:\.\w+)+) import \(?([\w ,\n]+)\)?", re.M)


def release_tests(manifest: dict, kept: Set[Path]) -> Tuple[List[Path], List[Path]]:
    """Split logic/test files into release tests and pruned tests (P-06)."""
    release: List[Path] = []
    pruned: List[Path] = []
    kept_modules = {file_to_module(f, REPO) for f in kept}
    for path in sorted((LOGIC / "test").rglob("*.py")):
        if "fixtures" in path.relative_to(LOGIC / "test").parts:
            # Fixture modules are judged by what they import: their mocker.patch target strings
            # may name excluded modules, and the tests that use those fixtures are validated by
            # running them in the staged tree.
            found = imports_of(path, REPO)
            names = {trimmed(n, REPO) for n in found.module_level | found.deferred}
            (release if {n for n in names if n} <= kept_modules else pruned).append(path)
            continue
        modules = set()
        text = path.read_text()
        for match in LOGIC_MODULE_RE.finditer(text):
            root = trimmed(match.group(0), REPO)
            if root:
                modules.add(root)
        # ``from <pkg> import <submodule>`` names a module the regex above sees as <pkg>.
        for match in FROM_IMPORT_RE.finditer(text):
            for name in re.findall(r"\w+", match.group(2)):
                if module_to_file(f"{match.group(1)}.{name}", REPO) is not None:
                    modules.add(f"{match.group(1)}.{name}")
        if modules <= kept_modules:
            release.append(path)
        else:
            pruned.append(path)
    return release, pruned
