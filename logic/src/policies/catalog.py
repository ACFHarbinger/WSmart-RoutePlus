"""Generated catalog of packing, routing, improvement, acceptance, operators, and models.

The Studio visualizer and builder read this document. It is not a hand-written
menu. Registered ids come from the stage manifests. Modules that exist on disk
and are not in a manifest are ``present``. A target that fails to import is
kept, with ``error`` set, so one broken module does not drop the rest.
"""

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from logic.src.policies.acceptance_criteria._manifest import ENTRIES as ACCEPTANCE_ENTRIES
from logic.src.policies.packing._manifest import ENTRIES as PACKING_ENTRIES
from logic.src.policies.route_improvement._manifest import ENTRIES as IMPROVEMENT_ENTRIES
from logic.src.policies.routing._manifest import ENTRIES as ROUTING_ENTRIES
from logic.src.utils.registry import import_target

_ROOT = Path(__file__).resolve().parents[1]
_POLICIES = _ROOT / "policies"

# Config ids the training hub launches. A target is set only when that class file is in the tree.
_SHIPPED_MODELS: Dict[str, Optional[str]] = {
    "am": "logic.src.models.core.attention_model:AttentionModel",
    "tam": "logic.src.models.core.temporal_attention_model.model:TemporalAttentionModel",
    "deep_decoder": "logic.src.models.core.attention_model.deep_decoder_policy:DeepDecoderPolicy",
    "moe": "logic.src.models.core.moe.moe_attention_model:MoEAttentionModel",
    "ptr": "logic.src.models.subnets.decoders.ptr.decoder:PointerDecoder",
    "symnco": "logic.src.models.core.attention_model.symnco_policy:SymNCOPolicy",
}

_PORTS: Dict[str, Dict[str, List[str]]] = {
    "packing": {"consumes": ["instance"], "produces": ["selected_nodes"]},
    "routing": {"consumes": ["instance"], "produces": ["routes"]},
    "route_improvement": {"consumes": ["routes"], "produces": ["routes"]},
    "acceptance": {"consumes": ["incumbent", "candidate"], "produces": ["decision"]},
    "operator": {"consumes": ["routes"], "produces": ["routes"]},
    "cut": {"consumes": ["lp_solution"], "produces": ["inequality"]},
    "model": {"consumes": ["instance"], "produces": ["routes"]},
    "reward_weight": {"consumes": ["reward"], "produces": ["weights"]},
}

_ALNS_STAGES = ["input", "initial_solution", "destroy", "repair", "acceptance", "incumbent", "output"]


def _try_import(target: Optional[str]) -> Optional[str]:
    if not target:
        return None
    try:
        import_target(target)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def _entry(
    entry_id: str,
    family: str,
    status: str,
    target: Optional[str],
    display: Optional[str] = None,
    stages: Optional[List[str]] = None,
    alias_of: Optional[str] = None,
    role: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    ports = _PORTS["reward_weight" if role == "reward_weight_adjustment" else family if family in _PORTS else "model"]
    if family == "separation":
        ports = _PORTS["cut"]
    document: Dict[str, Any] = {
        "id": entry_id,
        "display_name": display or entry_id,
        "family": family,
        "status": status,
        "target": target,
        "stages": list(stages or []),
        "consumes": list(ports["consumes"]),
        "produces": list(ports["produces"]),
    }
    if alias_of is not None:
        document["alias_of"] = alias_of
    if role is not None:
        document["role"] = role
    if error is not None:
        document["error"] = error
    return document


def _wants_resolve(resolve: bool, resolve_ids: Optional[Set[str]], name: str) -> bool:
    if not resolve:
        return False
    return resolve_ids is None or name in resolve_ids


def _from_manifest(
    entries: Dict[str, str],
    family: str,
    resolve: bool,
    canonical: Optional[Dict[str, str]] = None,
    resolve_ids: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """One catalog row per manifest id. Aliases point at the canonical id."""
    by_target: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    source = canonical if canonical is not None else entries
    for name, target in source.items():
        by_target.setdefault(target, name)
    for name, target in entries.items():
        alias = by_target.get(target)
        alias_of = alias if alias is not None and alias != name else None
        stages = _ALNS_STAGES if name == "alns" else ["input", "output"]
        error = _try_import(target) if _wants_resolve(resolve, resolve_ids, name) else None
        row = _entry(name, family, "registered", target, stages=stages, alias_of=alias_of, error=error)
        if name != "alns":
            row["interior"] = "not_instrumented"
        rows.append(row)
    return rows


def _python_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    )


def _present_acceptance(registered_targets: Dict[str, str]) -> List[Dict[str, Any]]:
    registered_stems = {target.split(":")[0].rsplit(".", 1)[-1] for target in registered_targets.values()}
    rows: List[Dict[str, Any]] = []
    directory = _POLICIES / "acceptance_criteria"
    for path in _python_files(directory):
        if path.stem in registered_stems:
            continue
        rows.append(
            _entry(
                path.stem,
                "acceptance",
                "present",
                f"logic.src.policies.acceptance_criteria.{path.stem}",
                stages=[],
            )
        )
    return rows


def _registered_packages(entries: Dict[str, str]) -> set:
    packages = set()
    for target in entries.values():
        module = target.split(":")[0]
        packages.add(module)
        parts = module.split(".")
        for index in range(1, len(parts)):
            packages.add(".".join(parts[:index]))
    return packages


def _present_routing(entries: Dict[str, str]) -> List[Dict[str, Any]]:
    """Algorithm packages under routing/ that no registered target imports."""
    covered = _registered_packages(entries)
    rows: List[Dict[str, Any]] = []
    routing = _POLICIES / "routing"
    for family_dir in sorted(path for path in routing.iterdir() if path.is_dir()):
        if family_dir.name in {"base", "__pycache__"}:
            continue
        for algo_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
            if algo_dir.name == "__pycache__":
                continue
            module = f"logic.src.policies.routing.{family_dir.name}.{algo_dir.name}"
            if module in covered:
                continue
            rows.append(
                _entry(
                    algo_dir.name,
                    "routing",
                    "present",
                    module,
                    stages=[],
                )
            )
            rows[-1]["routing_family"] = family_dir.name
            rows[-1]["interior"] = "not_instrumented"
    return rows


def _operator_families() -> List[Dict[str, Any]]:
    root = _POLICIES / "helpers" / "operators"
    rows: List[Dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for family_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "__pycache__"):
        names = [path.stem for path in _python_files(family_dir)]
        if not names and not (family_dir / "__init__.py").is_file():
            continue
        row = _entry(
            family_dir.name,
            "operator",
            "present",
            f"logic.src.policies.helpers.operators.{family_dir.name}",
            display=family_dir.name,
        )
        row["operators"] = names
        rows.append(row)
    return rows


def _separation() -> List[Dict[str, Any]]:
    directory = _POLICIES / "helpers" / "solvers_and_matheuristics" / "separation"
    rows: List[Dict[str, Any]] = []
    for path in _python_files(directory):
        rows.append(
            _entry(
                path.stem,
                "separation",
                "present",
                f"logic.src.policies.helpers.solvers_and_matheuristics.separation.{path.stem}",
            )
        )
    return rows


def _models(resolve: bool, resolve_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, target in _SHIPPED_MODELS.items():
        error = _try_import(target) if target and _wants_resolve(resolve, resolve_ids, name) else None
        row = _entry(
            name,
            "model",
            "shipped",
            target,
            stages=["input", "encoder", "decoder", "output"],
            role="routing_model",
            error=error,
        )
        rows.append(row)
    rows.append(
        _entry(
            "WeightAdjustmentRNN",
            "model",
            "present",
            "logic.src.models.meta.weight_adjustment_rnn:WeightAdjustmentRNN",
            role="reward_weight_adjustment",
            stages=[],
        )
    )
    core = _ROOT / "models" / "core"
    if core.is_dir():
        shipped_modules = {target.split(":")[0] for target in _SHIPPED_MODELS.values() if target}
        for algo_dir in sorted(path for path in core.iterdir() if path.is_dir() and path.name != "__pycache__"):
            module = f"logic.src.models.core.{algo_dir.name}"
            if any(module in item or item.startswith(module + ".") for item in shipped_modules):
                continue
            rows.append(_entry(algo_dir.name, "model", "present", module, role="routing_model", stages=[]))
    return rows


def build_catalog(resolve: bool = False, resolve_ids: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Return the catalog. ``resolve`` imports registered targets.

    ``resolve_ids`` limits that import to those ids. Import failures are stored
    on the entry. They do not raise.
    """
    # Touch the registry modules so a test that declared an extra name is visible.
    importlib.import_module("logic.src.policies.packing.base.registry")
    importlib.import_module("logic.src.policies.routing.base.registry")
    importlib.import_module("logic.src.policies.route_improvement.base.registry")
    importlib.import_module("logic.src.policies.acceptance_criteria.base.registry")

    from logic.src.utils.registry import ACCEPTANCE, IMPROVEMENT, PACKING, ROUTING

    def _with_live(manifest: Dict[str, str], registry: Any) -> Dict[str, str]:
        merged = dict(manifest)
        for name in registry.names():
            spec = registry._specs.get(name)
            if spec is None or name in merged:
                continue
            if spec.target:
                merged[name] = spec.target
        return merged

    packing = _with_live(dict(PACKING_ENTRIES), PACKING)
    routing = _with_live(dict(ROUTING_ENTRIES), ROUTING)
    improvement = _with_live(dict(IMPROVEMENT_ENTRIES), IMPROVEMENT)
    acceptance = _with_live(dict(ACCEPTANCE_ENTRIES), ACCEPTANCE)

    return {
        "schema_version": 1,
        "packing": _from_manifest(
            packing, "packing", resolve, canonical=dict(PACKING_ENTRIES), resolve_ids=resolve_ids
        ),
        "routing": _from_manifest(
            routing, "routing", resolve, canonical=dict(ROUTING_ENTRIES), resolve_ids=resolve_ids
        )
        + _present_routing(ROUTING_ENTRIES),
        "route_improvement": _from_manifest(
            improvement,
            "route_improvement",
            resolve,
            canonical=dict(IMPROVEMENT_ENTRIES),
            resolve_ids=resolve_ids,
        ),
        "acceptance": _from_manifest(
            acceptance, "acceptance", resolve, canonical=dict(ACCEPTANCE_ENTRIES), resolve_ids=resolve_ids
        )
        + _present_acceptance(ACCEPTANCE_ENTRIES),
        "operators": _operator_families(),
        "separation": _separation(),
        "models": _models(resolve, resolve_ids),
    }


def catalog_ids(catalog: Dict[str, Any], section: str) -> List[str]:
    """Ids in one catalog section, in file order."""
    return [str(row["id"]) for row in catalog[section]]


def missing_manifest_ids(catalog: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Manifest ids that the catalog dropped. Empty when the floor is intact."""
    expected = {
        "packing": PACKING_ENTRIES,
        "routing": ROUTING_ENTRIES,
        "route_improvement": IMPROVEMENT_ENTRIES,
        "acceptance": ACCEPTANCE_ENTRIES,
    }
    missing: List[Tuple[str, str]] = []
    for section, entries in expected.items():
        present = set(catalog_ids(catalog, section))
        for name in entries:
            if name not in present:
                missing.append((section, name))
    return missing
