"""Policy configurations of the benchmark campaign.

The campaign fixes the packing variants, the route improver and the routing limit of every
policy without editing the canonical policy files. For each routing algorithm and improver it
writes a patched copy of the policy YAML into an overlay directory, which a simulator run reads
through ``WSR_POLICY_CONFIG_DIR``.

Attributes:
    ROUTING_POLICIES: Routing algorithms of the campaign (policy file stems).
    STATIC_POLICY: The fixed-schedule baseline, run with the plain Regular rule.
    IMPROVERS: Route improver stage files by short name.
    CAMPAIGN_PACKING: Packing variants of the dynamic policies.
    DELAYED_PACKING: Threshold-regular variants whose calendar starts on day k.
    DELAYED_STATIC_PACKING: Fixed-schedule variants whose calendar starts on day k.
    ROUTING_LIMIT_S: Routing time limit of one daily solve, in seconds.
    patch_policy: Policy tree with the campaign packing, improver and limits.
    set_profit_aware: Policy tree with every profit_aware_operators switch set.
    campaign_packing_subset: The campaign packing variants restricted to some names.
    write_overlay: Patched policy file for one algorithm and improver.

Example:
    >>> patch_policy({"alns": {"custom": {"time_limit": 600.0, "packing": {}}}}, "cls")["alns"]["custom"]["time_limit"]
    60.0
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from logic.src.utils.configs.policy_files import find_policy_file

ROUTING_POLICIES = ("aco_hh", "pg_clns", "bpc", "alns", "swc_tcf", "hgs", "psoma", "sans_new")
# Exact methods whose plan-substituting fallbacks are switched off, so a day without an incumbent reports nothing.
STRICT_POLICIES = ("bpc", "swc_tcf")

STATIC_POLICY = "cvrp"
ROUTING_LIMIT_S = 60.0

# ``none`` leaves the routed tours as the algorithm returns them, to measure what improvement adds.
IMPROVERS: Dict[str, Any] = {
    "cls": {"other/route_improvement/ri_cls.yaml": ["default"]},
    "none": [],
}

CAMPAIGN_PACKING: List[Dict[str, List[str]]] = [
    {"other/packing/pk_last_minute.yaml": ["last_minute_cf90"]},
    {"other/packing/pk_service_level.yaml": ["service_level1"]},
    {"other/packing/pk_lookahead.yaml": ["lookahead"]},
    {"other/packing/pk_regular.yaml": ["regular_tr_lvl3", "regular_tr_lvl4", "regular_tr_lvl5"]},
]

# Delayed-start rerun: only the calendar rules depend on the start day, so the fill-level rules are not rerun.
DELAYED_PACKING: List[Dict[str, List[str]]] = [
    {
        "other/packing/pk_regular.yaml": [
            "regular_tr_lvl3_delayed",
            "regular_tr_lvl4_delayed",
            "regular_tr_lvl5_delayed",
        ]
    },
]
DELAYED_STATIC_PACKING: List[Dict[str, List[str]]] = [
    {"other/packing/pk_regular.yaml": ["regular_lvl3_delayed", "regular_lvl4_delayed", "regular_lvl5_delayed"]},
]


def patch_policy(
    tree: Any, improver: str, packing: Any = None, limit: float = ROUTING_LIMIT_S, strict: bool = False
) -> Any:
    """Apply the campaign settings to a policy configuration tree.

    Packing and route improvement are replaced wherever a policy declares them, ``time_limit``
    never exceeds ``limit``, and a day with no incumbent is not re-solved with a longer limit.

    Args:
        tree: Parsed policy YAML.
        improver: Key of ``IMPROVERS``.
        packing: Packing variants to use (default ``CAMPAIGN_PACKING``).
        limit: Routing limit in seconds.
        strict: Also set ``allow_fallback: false`` beside every routing time limit and use Gurobi's
            default feasibility tolerance (``feasibility_tol: -1``) where a policy has a ``formulation``.

    Returns:
        A patched deep copy of ``tree``.

    Raises:
        ValueError: If ``improver`` is not a known improver.
    """
    if improver not in IMPROVERS:
        raise ValueError(f"improver must be one of {sorted(IMPROVERS)}, got {improver!r}")
    chosen = copy.deepcopy(CAMPAIGN_PACKING if packing is None else packing)
    stage = copy.deepcopy(IMPROVERS[improver])

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key in list(node):
                if key == "packing":
                    node[key] = copy.deepcopy(chosen)
                elif key == "route_improvement":
                    node[key] = copy.deepcopy(stage)
                elif key == "time_limit" and isinstance(node[key], (int, float)):
                    node[key] = min(float(node[key]), float(limit))
                    if strict:
                        node["allow_fallback"] = False
                        if "greedy_warm_start" in node:  # BPC: an integer restricted master when the root is cut off
                            node["salvage_incomplete_node"] = True
                elif key == "formulation" and strict:
                    node["feasibility_tol"] = -1.0
                elif key == "no_solution_time_factor":
                    node[key] = 1.0
                else:
                    walk(node[key])
        elif isinstance(node, list):
            for item in node:
                walk(item)
        return node

    return walk(copy.deepcopy(tree))


def set_profit_aware(tree: Any, value: bool) -> int:
    """Set every ``profit_aware_operators`` key of a policy tree (in place).

    Args:
        tree: Parsed policy YAML.
        value: The switch value.

    Returns:
        int: Number of keys set.
    """
    count = 0
    if isinstance(tree, dict):
        for key in tree:
            if key == "profit_aware_operators":
                tree[key] = bool(value)
                count += 1
            else:
                count += set_profit_aware(tree[key], value)
    elif isinstance(tree, list):
        for item in tree:
            count += set_profit_aware(item, value)
    return count


def campaign_packing_subset(names: List[str]) -> List[Dict[str, List[str]]]:
    """The campaign packing variants whose names are in ``names``, file by file.

    Args:
        names: Variant names such as ``last_minute_cf90`` or ``regular_tr_lvl4``.

    Returns:
        List of ``{packing file: [variants]}`` entries.

    Raises:
        ValueError: If a name is not a campaign packing variant.
    """
    known = {v for entry in CAMPAIGN_PACKING for variants in entry.values() for v in variants}
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(f"not campaign packing variants: {unknown}; choose from {sorted(known)}")
    out = []
    for entry in CAMPAIGN_PACKING:
        for file, variants in entry.items():
            keep = [v for v in variants if v in names]
            if keep:
                out.append({file: keep})
    return out


def _set_model_path(node: Any, model_path: str) -> None:
    """Point every ``model_path`` key of a policy tree at ``model_path``."""
    if isinstance(node, dict):
        for key in node:
            if key == "model_path":
                node[key] = model_path
            else:
                _set_model_path(node[key], model_path)
    elif isinstance(node, list):
        for item in node:
            _set_model_path(item, model_path)


def write_overlay(
    policy: str,
    improver: str,
    out_dir: Path,
    packing: Any = None,
    canonical_dir: Path | None = None,
    model_path: str | None = None,
    profit_aware: bool | None = None,
) -> Path:
    """Write the patched policy file of one algorithm and improver.

    Args:
        policy: Policy name; a variant suffix such as ``sans_new`` selects its file by prefix.
        improver: Key of ``IMPROVERS``.
        out_dir: Overlay directory (created if missing).
        packing: Packing variants (default ``CAMPAIGN_PACKING``).
        canonical_dir: Policy configuration root (default ``logic/configs/policies``).
        model_path: Trained model directory that replaces every ``model_path`` of a learned policy.
        profit_aware: When given, the value of every ``profit_aware_operators`` switch (the ablation arms).

    Returns:
        Path: The written file, named like the canonical one.

    Raises:
        FileNotFoundError: If no policy file matches ``policy``.
    """
    from logic.src import constants as udef

    root = canonical_dir or Path(udef.CONFIGS_DIR) / "policies"
    stem = policy
    source = None
    while stem and source is None:
        source = find_policy_file(f"policy_{stem}.yaml", root)
        stem = stem.rpartition("_")[0] if source is None else stem
    if source is None:
        raise FileNotFoundError(f"no policy file for {policy!r} under {root}")
    patched = patch_policy(
        yaml.safe_load(Path(source).read_text()), improver, packing, strict=policy in STRICT_POLICIES
    )
    if model_path is not None:
        _set_model_path(patched, model_path)
    if profit_aware is not None and set_profit_aware(patched, profit_aware) == 0:
        raise ValueError(f"{policy!r} has no profit_aware_operators switch")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(source).name
    target.write_text(yaml.dump(patched, default_flow_style=False, sort_keys=False))
    return target
