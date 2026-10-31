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
    ROUTING_LIMIT_S: Routing time limit of one daily solve, in seconds.
    patch_policy: Policy tree with the campaign packing, improver and limits.
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

IMPROVERS: Dict[str, Dict[str, List[str]]] = {
    "cls": {"other/route_improvement/ri_cls.yaml": ["default"]},
    "ftsp": {"other/route_improvement/ri_ftsp.yaml": ["ftsp"]},
}

CAMPAIGN_PACKING: List[Dict[str, List[str]]] = [
    {"other/packing/pk_last_minute.yaml": ["last_minute_cf90"]},
    {"other/packing/pk_service_level.yaml": ["service_level1"]},
    {"other/packing/pk_lookahead.yaml": ["lookahead"]},
    {"other/packing/pk_regular.yaml": ["regular_tr_lvl3", "regular_tr_lvl4", "regular_tr_lvl5"]},
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


def write_overlay(
    policy: str, improver: str, out_dir: Path, packing: Any = None, canonical_dir: Path | None = None
) -> Path:
    """Write the patched policy file of one algorithm and improver.

    Args:
        policy: Policy name; a variant suffix such as ``sans_new`` selects its file by prefix.
        improver: Key of ``IMPROVERS``.
        out_dir: Overlay directory (created if missing).
        packing: Packing variants (default ``CAMPAIGN_PACKING``).
        canonical_dir: Policy configuration root (default ``logic/configs/policies``).

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
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(source).name
    target.write_text(yaml.dump(patched, default_flow_style=False, sort_keys=False))
    return target
