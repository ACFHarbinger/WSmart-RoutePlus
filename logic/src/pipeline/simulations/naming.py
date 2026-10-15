"""Policy display names and result keys.

Re-exported from ``day_context`` so existing callers keep their import path.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _registered_constructor_keys() -> Tuple[str, ...]:
    """Return every routing-algorithm key, taken from ``logic/configs/policies/policy_<key>.yaml``.

    The file names are used instead of the live ``RoutingAlgorithmRegistry`` because the registry
    only holds the adapters imported so far (SWC-TCF, for one, is imported lazily), which would
    make the seed depend on import order.
    """
    from logic.src.constants.paths import CONFIGS_DIR

    policies_dir = os.path.join(CONFIGS_DIR, "policies")
    try:
        names = os.listdir(policies_dir)
    except OSError:  # pragma: no cover - frozen builds without configs fall back to the token heuristic
        return ()
    return tuple(
        sorted(n[len("policy_") : -len(".yaml")] for n in names if n.startswith("policy_") and n.endswith(".yaml"))
    )


# Tokens that name a selector or an improvement stage, not a routing algorithm.
_STRUCTURAL_NAME_TOKENS = frozenset({"ms", "ri", "none", "custom", "new", "og"})


def _matching_constructor(parts: List[str]) -> Optional[str]:
    """Longest registered routing algorithm key inside an expanded policy id.

    ``alns`` and ``hgs`` are routing algorithms, so they are not skipped. Selection
    tokens such as ``regular`` are not routing algorithm keys and do not win.

    Args:
        parts: Underscore tokens of the policy id.

    Returns:
        The routing algorithm key, or None when the registry has no match.
    """
    keys = set(_registered_constructor_keys())
    for width in range(len(parts), 0, -1):
        for start in range(len(parts) - width + 1):
            candidate = "_".join(parts[start : start + width])
            if candidate in keys and candidate not in _STRUCTURAL_NAME_TOKENS:
                return candidate
    return None


def get_canonical_policy_name(policy_name: str, known_keys: Optional[Iterable[str]] = None) -> str:
    """
    Get the routing-algorithm key of a policy slug, used for RNG seeding.

    The seed must depend on the routing algorithm only, so that the same routing algorithm
    under different selectors or thresholds draws the same random stream, and
    different routing algorithms under the same selector do not. The routing algorithm is the
    leftmost-longest run of slug tokens that is a registered routing algorithm key.

    Args:
        policy_name: Full policy slug (e.g., 'last_minute_cf70_aco_hh_custom_cls').
        known_keys: Routing algorithm keys to match; defaults to the ``policy_<key>.yaml`` names.

    Returns:
        The routing algorithm key (e.g., 'aco_hh'). If no key matches, the slug with any
        leading selection tokens removed is returned.

    Examples:
        >>> get_canonical_policy_name('last_minute_cf70_aco_hh_custom_cls', ['aco_hh', 'alns'])
        'aco_hh'
        >>> get_canonical_policy_name('lookahead_ma_ts_custom_gamma3', ['ma', 'ma_ts'])
        'ma_ts'
    """
    parts = policy_name.lower().split("_")
    keys = set(known_keys) if known_keys is not None else set(_registered_constructor_keys())
    for width in range(len(parts), 0, -1):
        for start in range(len(parts) - width + 1):
            candidate = "_".join(parts[start : start + width])
            if candidate in keys:
                return candidate

    selection_tokens = {"lookahead", "policy", "regular", "last", "minute", "lastminute", "revenue", "service"}
    rest = [p for p in parts if p not in selection_tokens and not p.startswith(("cf", "level"))]
    return "_".join(rest) if rest else policy_name.lower()


def _clean(name: Any) -> str:  # noqa: C901
    """Clean a policy/strategy name for display."""
    if name is None:
        return "None"

    # Handle lists/sequences (including OmegaConf ListConfig and nested lists)
    # Recursively unwrap until we find a string or a dict/config
    if not isinstance(name, str) and hasattr(name, "__iter__") and not hasattr(name, "items"):
        try:
            name_list = list(name)
            if len(name_list) > 0:
                # Try each item until we find something valid
                for item in name_list:
                    cleaned = _clean(item)
                    if cleaned != "None":
                        return cleaned
            return "None"
        except Exception:
            return "None"

    # Handle string names/paths
    if isinstance(name, str):
        if name.lower() in ["none", "null", "false", "[]", "none.yaml"]:
            return "None"

        # If it's a full path, get the basename
        base = os.path.basename(name)
        # Remove common prefixes and extensions
        for p in ["ms_", "ri_", "ac_", "policy_", ".yaml", ".xml", ".json"]:
            base = base.replace(p, "")

        # Clean up any remaining directory parts if basename failed (e.g. windows paths)
        if "/" in base:
            base = base.split("/")[-1]
        if "\\" in base:
            base = base.split("\\")[-1]

        return base.replace("_", " ").title()

    # Handle structured config objects (Dataclasses, Dicts, DictConfigs)
    if hasattr(name, "strategy"):
        return _clean(name.strategy)

    if hasattr(name, "get") or isinstance(name, Mapping):
        # Handle new dict format: {"other/pk_last_minute.yaml": "cf70"}
        if len(name) == 1:
            key, val = next(iter(name.items()))
            key_str = str(key)
            if key_str.endswith(".yaml") or key_str.endswith(".xml"):
                raw_v = val
                if hasattr(val, "__iter__") and not isinstance(val, str):
                    try:
                        v_list = list(val)
                        raw_v = v_list[0] if v_list else ""
                    except Exception:
                        pass
                variant = str(raw_v) if raw_v is not None else ""
                if variant and variant != "default":
                    return variant.replace("_", " ").title()
                base = os.path.basename(key_str)
                for p in ["ms_", "ri_", "ac_", "policy_", ".yaml", ".xml", ".json"]:
                    base = base.replace(p, "")
                return base.replace("_", " ").title()

        # Check exhaustive list of common keys for strategies
        for k in ["strategy", "type", "name", "id", "model", "method"]:
            val = name.get(k) if hasattr(name, "get") else name.get(k, None)
            if val:
                cleaned = _clean(val)
                if cleaned != "None":
                    return cleaned
        return "Custom"

    return "None"


# Recursive helper to find naming keys in potentially nested/listed configs
def find_policy_keys(obj: Any) -> Dict[str, Any]:
    """
    Recursively find packing and route_improvement keys in a config.
    Searches for synonyms and handles deep nesting.

    Args:
        obj: The configuration object (dict, list, DictConfig, etc.)

    Returns:
        Dictionary containing the found keys.
    """
    found = {}

    # Synonyms for metadata keys
    MS_KEYS = {"packing", "ms", "node_selection", "selection_strategy"}
    RI_KEYS = {"route_improvement", "ri", "improvement_strategy", "local_search"}
    AC_KEYS = {"acceptance_criteria", "acceptance_criterion", "ac"}

    # Handle Mapping (dict, DictConfig)
    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        # 1. Check current level for direct matches or synonyms
        for key, value in obj.items():
            if key in MS_KEYS:
                if value is not None and str(value).lower() not in ["none", "null", "false", "[]"]:
                    found["packing"] = value
            elif key in RI_KEYS and value is not None and str(value).lower() not in ["none", "null", "false", "[]"]:
                found["route_improvement"] = value
            elif key in AC_KEYS and value is not None and str(value).lower() not in ["none", "null", "false", "[]"]:
                found["acceptance_criteria"] = value

        # 2. Recurse into all values to find deeper configurations
        # We process values in order, so deeper values will overwrite shallower ones
        for v in obj.values():
            if v is not None and not isinstance(v, (str, int, float, bool)):
                inner = find_policy_keys(v)
                if inner:
                    found.update(inner)

    # Handle sequences (list, ListConfig)
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        for item in obj:
            if item is not None and not isinstance(item, (str, int, float, bool)):
                inner = find_policy_keys(item)
                if inner:
                    found.update(inner)

    return found


def build_naming_config(pol_cfg: Any, global_sim_cfg: Any) -> Dict[str, Any]:
    """
    Build a configuration dictionary specifically for naming resolution.
    It combines keys found in the policy config with global simulation defaults.

    Args:
        pol_cfg: The policy-specific configuration.
        global_sim_cfg: The global simulation configuration (ctx.cfg.sim).

    Returns:
        A dictionary with 'packing' and 'route_improvement' keys.
    """
    # 1. Extract keys from policy config (recursive)
    extracted = find_policy_keys(pol_cfg)

    # 2. Fallback to global simulation defaults if keys are missing or None
    for key in ["packing", "route_improvement", "acceptance_criteria"]:
        if not extracted.get(key) and hasattr(global_sim_cfg, key):
            val = getattr(global_sim_cfg, key)
            if val:
                extracted[key] = val

    return extracted


def resolve_policy_display_name(policy: Any, sim_cfg: Any) -> Tuple[str, str]:
    """Resolves the original policy ID and its full descriptive display name.

    Args:
        policy: The policy configuration (string or dict).
        sim_cfg: The simulation configuration.

    Returns:
        Tuple of (pol_id_orig, display_name)
    """
    from logic.src.utils.configs.config_loader import load_config
    from logic.src.utils.infrastructure.setup_sims import deep_sanitize, get_pol_name

    pol_id_orig = get_pol_name(policy)
    sanitized_policy = deep_sanitize(policy)
    pol_cfg = {}

    if isinstance(sanitized_policy, str):
        config_paths = deep_sanitize(sim_cfg.config_path) if hasattr(sim_cfg, "config_path") else {}
        if config_paths is None:
            config_paths = {}
        if pol_id_orig in config_paths:
            loaded_cfg = deep_sanitize(config_paths[pol_id_orig])
            if isinstance(loaded_cfg, dict):
                pol_cfg = loaded_cfg
            elif isinstance(loaded_cfg, str) and os.path.exists(loaded_cfg):
                try:
                    pol_cfg = load_config(loaded_cfg)
                except Exception:
                    pol_cfg = {}
    elif isinstance(sanitized_policy, dict) and len(sanitized_policy) == 1 and pol_id_orig in sanitized_policy:
        pol_cfg = sanitized_policy[pol_id_orig]
    elif isinstance(sanitized_policy, dict):
        pol_cfg = sanitized_policy

    sanitized_pol_cfg = deep_sanitize(pol_cfg)
    naming_cfg = build_naming_config(sanitized_pol_cfg, sim_cfg)
    display_name = get_full_policy_name(pol_id_orig, naming_cfg)

    return pol_id_orig, display_name


def policy_result_key(policy: Any, sim: Any) -> str:
    """Key a policy's results are stored and looked up under.

    This is the slug of the display name, the same string used for ``ctx.pol_name``,
    the ``log_<key>_<N>N.json`` file name and the realtime jsonl. The expander id
    (``get_pol_name``) differs from it (e.g. ``..._emp`` vs ``..._none``) and must not
    be used for lookups.

    Args:
        policy: Entry of ``sim.full_policies``.
        sim: Simulation config.

    Returns:
        The result key.
    """
    return to_slug(resolve_policy_display_name(policy, sim)[1])


def to_slug(name: str) -> str:
    """Converts a display name to a safe slug string for IDs and filenames.

    Strips parenthetical distribution tags (e.g. " (Gamma3)", " (Emp)") so
    they appear in human-readable output but never reach the log filename.
    Replaces ' + ' with '_', removes remaining '()' chars, and lowercases.

    Args:
        name: The display name to convert.

    Returns:
        The slug string.
    """
    # Strip " (Gamma3)"-style parenthetical suffixes before building the slug
    name = re.sub(r"\s*\([^)]*\)", "", name)
    return name.replace(" + ", "_").replace(" ", "_").replace("(", "").replace(")", "").lower()


def get_full_policy_name(pol_name: str, config: Dict[str, Any]) -> str:
    """
    Construct a descriptive policy name including packing and route improvement.

    This reflects the entire construction and improvement stack.

    Args:
        pol_name: Base policy name (the routing algorithm).
        config: The naming configuration dictionary.

    Returns:
        A formatted name string (e.g., 'Lookahead + AM + None').
    """

    # 1. Extract Packing
    ms = config.get("packing")
    ms_name = _clean(ms)

    # 2. Extract Routing (Base Policy)
    # Strip common prefixes/suffixes
    base_id = pol_name.lower().replace("policy_", "")

    # Remove distribution suffix and anything after it
    for suffix in ["_emp", "_gamma"]:
        if suffix in base_id:
            base_id = base_id.split(suffix)[0]

    if "_" in base_id:
        # If it looks expanded (e.g., 'ms_regular_alns_ri_none'), try to extract the middle
        parts = base_id.split("_")
        if "ms" in parts or "ri" in parts:
            matched = _matching_constructor(parts)
            if matched is not None:
                base_id = matched
            else:
                for part in parts:
                    if part not in _STRUCTURAL_NAME_TOKENS:
                        base_id = part
                        break

    # 3. Extract Route Improvement
    ri = config.get("route_improvement")
    ri_name = _clean(ri)

    # Remove redundant packing name from base_id to prevent duplication
    ms_slug = ms_name.lower().replace(" ", "_")
    if ms_slug != "none" and ms_slug in base_id:
        base_id = base_id.replace(f"_{ms_slug}", "").replace(f"{ms_slug}_", "").replace(ms_slug, "")

    # 4. Extract Acceptance Criteria
    ac = config.get("acceptance_criteria")
    ac_name = _clean(ac)

    # Remove redundant route improvement name from base_id to prevent duplication
    ri_slug = ri_name.lower()
    if ri_slug != "none" and ri_slug in base_id:
        base_id = base_id.replace(f"_{ri_slug}", "").replace(f"{ri_slug}_", "").replace(ri_slug, "")

    # Remove redundant acceptance criteria name from base_id to prevent duplication
    ac_slug = ac_name.lower()
    if ac_slug != "none" and ac_slug in base_id:
        base_id = base_id.replace(f"_{ac_slug}", "").replace(f"{ac_slug}_", "").replace(ac_slug, "")

    # Extract distribution suffix for display only (most specific first)
    dist_tag = ""
    for suffix in ["_gamma3", "_gamma2", "_gamma1", "_gamma", "_emp"]:
        if suffix in pol_name.lower():
            dist_tag = f" ({suffix.lstrip('_').capitalize()})"
            break

    base_name = base_id.upper()
    if ac_name.lower() != "none":
        return f"{ms_name} + {base_name} + {ac_name} + {ri_name}{dist_tag}"
    else:
        return f"{ms_name} + {base_name} + {ri_name}{dist_tag}"


