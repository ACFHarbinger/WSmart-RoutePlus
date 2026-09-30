"""
Base classes and utilities for simulation actions.

Attributes:
    SimulationAction: Abstract base class for simulation actions.

Example:
    >>> # from logic.src.pipeline.simulations.actions.base import SimulationAction
    >>> # class MyAction(SimulationAction):
    >>> #     def execute(self, context): ...
"""

import json
import os
from abc import ABC, abstractmethod
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple


def _find_key(d: Any, target_key: str) -> Any:
    """
    Recursively find the first occurrence of a target key in a nested structure.
    Handles dictionaries, ITraversable objects, and iterable sequences (lists, ListConfig, etc.).

    Args:
        d: The nested structure to search within.
        target_key: The key to search for.

    Returns:
        The value of the first target key found, or None if not found.
    """
    if isinstance(d, dict) or hasattr(d, "items"):
        d_dict = dict(d) if hasattr(d, "items") else d
        if target_key in d_dict:
            return d_dict[target_key]
        for val in d_dict.values():
            res = _find_key(val, target_key)
            if res is not None:
                return res
    elif isinstance(d, (list, tuple)) or (
        not isinstance(d, (str, dict)) and hasattr(d, "__iter__")
    ):  # Handle sequences including ListConfig
        for item in d:
            res = _find_key(item, target_key)
            if res is not None:
                return res
    return None


_EXPANDED_KEYS = ("mandatory_selection", "route_improvement", "acceptance_criteria", "acceptance_criterion")


def _flatten_config(cfg: Any) -> dict:  # noqa: C901
    """
    Helper to flatten nested configuration structures (e.g. hgs.custom -> list of dicts).
    Args:
        cfg: The configuration object to flatten.

    Returns:
        A flattened dictionary of the configuration.
    """
    if not cfg:
        return {}

    curr = cfg

    # Use a type guard to help Pyright/Pyrefly
    while isinstance(curr, Mapping) and len(curr) == 1:
        # Safer key extraction
        key = next(iter(curr.keys()))

        if key in ["mandatory_selection", "policy", "route_improvement", "acceptance_criterion", "acceptance_criteria"]:
            break
        curr = curr[key]

    # Handle list of dicts (including OmegaConf ListConfig)
    if isinstance(curr, (list, tuple)) or (not isinstance(curr, (str, dict, Mapping)) and hasattr(curr, "__iter__")):
        merged: Dict[str, Any] = {}
        for item in curr:
            if isinstance(item, Mapping):
                merged.update(item)  # No need to cast to dict() if it's already a Mapping
        return merged

    # Handle dict/Mapping
    if isinstance(curr, Mapping):
        flat = dict(curr)
        # Iterate over all keys and flatten if value is a list of dicts
        for _k, v in list(flat.items()):
            if isinstance(v, (list, tuple)) or (not isinstance(v, (str, Mapping)) and hasattr(v, "__iter__")):
                primitive_list = []
                for item in v:
                    if hasattr(item, "items"):
                        for sub_k, sub_v in dict(item).items():
                            # Keep the variant-specific selection/improvement/acceptance entries that
                            # the policy expander already resolved at the top level; nested 'custom'
                            # blocks still carry the unexpanded {file: [all variants]} form.
                            if sub_k in _EXPANDED_KEYS and sub_k in flat:
                                continue
                            flat[sub_k] = sub_v
                    else:
                        primitive_list.append(item)
                if primitive_list:
                    flat[_k] = primitive_list

        for target in ["mandatory_selection", "route_improvement", "acceptance_criterion", "acceptance_criteria"]:
            if target not in flat:
                val = _find_key(curr, target)
                if val is not None:
                    flat[target] = val

        return flat

    return {}


def _is_mapping(obj: Any) -> bool:
    """True for dict / DictConfig, not for strings, lists, or ListConfig."""
    if isinstance(obj, dict):
        return True
    if isinstance(obj, (str, list, tuple, bytes)):
        return False
    try:
        from omegaconf import DictConfig, ListConfig

        if isinstance(obj, ListConfig):
            return False
        if isinstance(obj, DictConfig):
            return True
    except Exception:
        pass
    return hasattr(obj, "items") and hasattr(obj, "keys") and not hasattr(obj, "append")


def _as_plain_dict(cfg: Any) -> Dict[str, Any]:
    """Shallow-convert a mapping (including OmegaConf) to a builtin dict."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "items"):
        return dict(cfg.items())
    return {}


def _is_non_string_sequence(obj: Any) -> bool:
    """True for list / tuple / ListConfig, not for mappings or strings."""
    if isinstance(obj, (str, bytes, dict)):
        return False
    if _is_mapping(obj):
        return False
    return isinstance(obj, (list, tuple)) or hasattr(obj, "__iter__")


def _unwrap_variant_value(val: Any) -> str:
    """Take a yaml variant that may be a string or a one-element list / ListConfig."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if _is_non_string_sequence(val):
        try:
            items = list(val)
        except TypeError:
            return str(val)
        return str(items[0]) if items else ""
    return str(val)


def _file_yaml_pairs(item: Any) -> List[Tuple[str, str]]:
    """Return ``[(file.yaml, variant), ...]`` for ``{file: variant}`` or ``{file: [v, ...]}``.

    Args:
        item: A config item that may be a file-to-variant mapping.

    Returns:
        A list of (relative yaml path, variant name) pairs. Empty if ``item`` is not
        in that form.
    """
    if not _is_mapping(item):
        return []
    try:
        item_dict = _as_plain_dict(item)
    except Exception:
        return []
    if len(item_dict) != 1:
        return []
    key, val = next(iter(item_dict.items()))
    if not isinstance(key, str) or not (key.endswith(".yaml") or key.endswith(".xml")):
        return []
    if isinstance(val, str) or val is None:
        return [(key, val or "")]
    if _is_non_string_sequence(val):
        variants = [str(v) for v in val]
        return [(key, v) for v in variants] if variants else [(key, "")]
    return [(key, _unwrap_variant_value(val))]


def _load_policy_yaml_section(file_key: str, variant: str) -> Dict[str, Any]:
    """Load ``logic/configs/policies/<file_key>`` and navigate into ``variant``.

    A ``default`` variant that is not a top-level key descends into the sole
    mapping (lookahead yaml is keyed ``lookahead:``, CLS yaml is keyed ``default:``).

    Args:
        file_key: Path relative to ``configs/policies``.
        variant: Named section, or ``default`` / empty to take the only child.

    Returns:
        The selected section as a plain dict (empty if the file cannot be read).
    """
    from logic.src.constants.paths import CONFIGS_DIR
    from logic.src.utils.configs.config_loader import load_config

    fpath = os.path.join(CONFIGS_DIR, "policies", file_key)
    try:
        cfg = load_config(fpath)
    except (OSError, ValueError):
        return {}
    if not cfg:
        return {}
    cfg_dict = _as_plain_dict(cfg)
    if "config" in cfg_dict and len(cfg_dict) == 1:
        cfg_dict = _as_plain_dict(cfg_dict["config"])
    variant = variant or ""
    if variant and variant in cfg_dict:
        inner = cfg_dict[variant]
        return _as_plain_dict(inner) if _is_mapping(inner) else {"value": inner}
    if len(cfg_dict) == 1:
        inner = next(iter(cfg_dict.values()))
        if _is_mapping(inner):
            return _as_plain_dict(inner)
    return cfg_dict


def _params_as_dict(params: Any) -> Dict[str, Any]:
    """Convert dataclass / mapping params to a plain dict."""
    if params is None:
        return {}
    if is_dataclass(params) and not isinstance(params, type):
        return asdict(params)
    if _is_mapping(params):
        return _as_plain_dict(params)
    if hasattr(params, "__dict__"):
        return {k: v for k, v in vars(params).items() if not k.startswith("_")}
    return {}


def _make_acceptance_config(method: str, params: Any) -> Any:
    """Build the typed AcceptanceConfig the constructors already consume.

    Args:
        method: Criterion name from yaml (``bmc``, ``oi``, ...).
        params: Mapping or dataclass of constructor kwargs.

    Returns:
        ``AcceptanceConfig`` with nested param dataclass for BMC / OI.
    """
    from logic.src.configs.policies.other.acceptance_criteria import (
        AcceptanceConfig,
        BoltzmannAcceptanceConfig,
        OnlyImprovingConfig,
    )

    method_l = str(method or "").lower()
    p = _params_as_dict(params)
    if method_l in ("bmc", "boltzmann", "boltzmann_metropolis", "boltzmann_metropolis_criterion"):
        return AcceptanceConfig(
            method="bmc",
            params=BoltzmannAcceptanceConfig(
                initial_temp=float(p.get("initial_temp", 100.0)),
                alpha=float(p.get("alpha", 0.995)),
                seed=int(p.get("seed", 42)),
            ),
        )
    if method_l in ("oi", "only_improving"):
        return AcceptanceConfig(method="oi", params=OnlyImprovingConfig())
    return AcceptanceConfig(method=method_l or "oi", params=p)


def _acceptance_config_from_entry(raw: Any) -> Optional[Any]:
    """Resolve ``acceptance_criteria`` yaml / mapping to a typed AcceptanceConfig.

    Args:
        raw: Policy-level ``acceptance_criteria`` value.

    Returns:
        AcceptanceConfig or None if nothing is configured.
    """
    if raw is None or raw is False:
        return None
    if isinstance(raw, str) and raw.lower() in ("none", "null", ""):
        return None
    if _is_non_string_sequence(raw):
        for item in raw:
            resolved = _acceptance_config_from_entry(item)
            if resolved is not None:
                return resolved
        return None
    pairs = _file_yaml_pairs(raw)
    if pairs:
        file_key, variant = pairs[0]
        section = _load_policy_yaml_section(file_key, variant)
        method = section.get("method") or variant or file_key
        return _make_acceptance_config(str(method), section.get("params") or {})
    if _is_mapping(raw):
        raw_dict = _as_plain_dict(raw)
        if "method" in raw_dict:
            return _make_acceptance_config(str(raw_dict.get("method")), raw_dict.get("params") or {})
    return None


def _inject_acceptance_into_config(cfg: Any) -> Optional[Any]:
    """Write yaml acceptance params onto ``acceptance_criterion`` in ``cfg``.

    Constructors read ``acceptance_criterion`` (singular) and ignore the
    policy-level ``acceptance_criteria: {file: [variant]}`` key. Injecting the
    typed object here is what makes PSOMA / ALNS / HGS honour ``ac_bmc.yaml``
    and ``ac_oi.yaml``.

    Args:
        cfg: Mutable policy config (dict). DictConfig should already be converted.

    Returns:
        The injected AcceptanceConfig, or None.
    """
    if not cfg:
        return None
    resolved = _acceptance_config_from_entry(_find_key(cfg, "acceptance_criteria"))
    if resolved is None:
        resolved = _acceptance_config_from_entry(_find_key(cfg, "acceptance_criterion"))
        if resolved is None:
            return None
        # Already a typed criterion / mapping with method — still ensure the
        # constructor field is present as AcceptanceConfig.
        if not hasattr(resolved, "method"):
            return None

    def _walk(node: Any) -> None:
        if _is_mapping(node):
            node_dict = node if isinstance(node, dict) else None
            if node_dict is not None:
                if "acceptance_criteria" in node_dict or "acceptance_criterion" in node_dict:
                    node_dict["acceptance_criterion"] = resolved
            else:
                if "acceptance_criteria" in node or "acceptance_criterion" in node:
                    node["acceptance_criterion"] = resolved
            values = node.values() if hasattr(node, "values") else []
            for value in values:
                _walk(value)
        elif _is_non_string_sequence(node):
            for item in node:
                _walk(item)

    _walk(cfg)
    return resolved


def _live_ac_payload(resolved: Any) -> Dict[str, Any]:
    """JSON-friendly snapshot of an AcceptanceConfig."""
    if resolved is None:
        return {}
    method = getattr(resolved, "method", None)
    params = _params_as_dict(getattr(resolved, "params", None))
    if method is None and _is_mapping(resolved):
        method = _as_plain_dict(resolved).get("method")
        params = _params_as_dict(_as_plain_dict(resolved).get("params"))
    return {"method": method, "params": params}


class _AttrDict(dict):
    """Mapping compatibility for consumers supporting attribute-style config access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _attrify(obj: Any) -> Any:
    """Recursively wrap dicts as ``_AttrDict``; leave dataclasses intact."""
    if isinstance(obj, _AttrDict):
        return _AttrDict({k: _attrify(v) for k, v in obj.items()})
    if isinstance(obj, dict):
        return _AttrDict({k: _attrify(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_attrify(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_attrify(v) for v in obj)
    return obj


def _ensure_mutable_config(context: Dict[str, Any]) -> Any:
    """Convert Hydra DictConfig policy blobs to a mutable mapping the day can mutate."""
    cfg = context.get("config", {})
    if isinstance(cfg, _AttrDict):
        return cfg
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        elif isinstance(cfg, dict):
            # Nested Hydra nodes stay DictConfig even when the top mapping is a dict.
            cfg = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
        cfg = _attrify(cfg)
        context["config"] = cfg
    except Exception:
        pass
    return cfg


def _pop_context_key(context: Any, key: str) -> Any:
    """Pop a key from a dict or ``SimulationDayContext`` (Mapping without ``pop``)."""
    if isinstance(context, dict):
        return context.pop(key, None)
    value = context.get(key) if hasattr(context, "get") else getattr(context, key, None)
    with suppress(Exception):
        context[key] = None
    return value


def _append_context_list(context: Any, key: str, item: Any) -> None:
    """Append to a list stored on a dict or day context (no ``setdefault``)."""
    current = context.get(key) if hasattr(context, "get") else None
    if not isinstance(current, list):
        current = []
        context[key] = current
    current.append(item)


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of dataclasses / numpy scalars for JSON capture."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


_LIVE_CAPTURE_META: ContextVar[Optional[Dict[str, Any]]] = ContextVar("wsr_live_capture_meta", default=None)
_PAPER_CONSTRUCTORS = (
    "ms_bpc_sp",
    "pg_clns",
    "swc_tcf",
    "aco_hh",
    "psoma",
    "alns",
    "hgs",
    "bpc",
    "na",
)


def _constructor_from_policy_id(policy_id: str) -> str:
    """Longest paper-constructor key inside an expanded policy id."""
    padded = f"_{policy_id}_"
    best = ""
    for key in _PAPER_CONSTRUCTORS:
        if f"_{key}_" in padded and len(key) > len(best):
            best = key
    return best


def _set_live_capture_meta(context: Any) -> None:
    """Remember policy/variant/day for subsequent consumer JSONL records."""
    policy_id = str(context.get("full_policy") or context.get("policy_name") or "")
    _LIVE_CAPTURE_META.set(
        {
            "policy": policy_id,
            "constructor": _constructor_from_policy_id(policy_id),
            "display_name": str(context.get("display_name") or ""),
            "day": context.get("day"),
            "sample_id": context.get("sample_id"),
        }
    )


def _record_live_params(kind: str, payload: Dict[str, Any]) -> None:
    """Append one JSONL record when ``WSR_CAPTURE_STRATEGY_JSON`` is set.

    Used by the live ``main.py test_sim`` check so captured MS / RI / AC
    parameters can be compared to yaml without constructing a policy from a
    plain dict.

    Args:
        kind: ``mandatory_selection``, ``route_improvement``, or ``acceptance_criteria``.
        payload: JSON-serialisable snapshot of the live parameters.
    """
    path = os.environ.get("WSR_CAPTURE_STRATEGY_JSON")
    if not path:
        return
    record: Dict[str, Any] = {"kind": kind, "payload": _jsonable(payload)}
    meta = _LIVE_CAPTURE_META.get() or {}
    for key, value in meta.items():
        if value not in (None, ""):
            record[key] = _jsonable(value)
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as handle:
        with suppress(Exception):
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(line)


class SimulationAction(ABC):
    """
    Abstract base class for simulation actions.

    Defines the interface for all simulation commands. Each action receives
    a shared context dictionary and modifies it in-place with its outputs.

    Attributes:
        None
    """

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> None:
        """
        Executes the action and updates the context in-place.

        Args:
            context: Shared dictionary containing simulation state.
        """
        pass
