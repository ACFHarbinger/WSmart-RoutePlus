"""
Base classes and utilities for simulation actions.

Attributes:
    SimulationAction: Abstract base class for simulation actions.

Example:
    >>> # from logic.src.pipeline.simulations.actions.base import SimulationAction
    >>> # class MyAction(SimulationAction):
    >>> #     def execute(self, context): ...

The policy-config mapping and acceptance-config helpers (``_find_key`` to
``_inject_acceptance_into_config`` and ``_live_ac_payload``) live in ``policy_config.py``
and are re-exported here.
"""

import json
import os
from abc import ABC, abstractmethod
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional


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
    cfg = context.get("config", {}) if isinstance(context, dict) else getattr(context, "config", {})
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
        if isinstance(context, dict):
            context["config"] = cfg
        else:
            context.config = cfg
    except Exception:
        pass
    return cfg


def _pop_context_key(context: Any, key: str) -> Any:
    """Clear a key on a dict or day context and return the previous value."""
    if isinstance(context, dict):
        return context.pop(key, None)
    value = getattr(context, key, None)
    with suppress(Exception):
        setattr(context, key, None)
    return value


def _append_context_list(context: Any, key: str, item: Any) -> None:
    """Append to a list stored on a dict or day context (no ``setdefault``)."""
    current = context.get(key) if isinstance(context, dict) else getattr(context, key, None)
    if not isinstance(current, list):
        current = []
        if isinstance(context, dict):
            context[key] = current
        else:
            setattr(context, key, current)
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

    def _field(name: str) -> Any:
        if isinstance(context, dict):
            return context.get(name)
        return getattr(context, name, None)

    policy_id = str(_field("full_policy") or _field("policy_name") or "")
    _LIVE_CAPTURE_META.set(
        {
            "policy": policy_id,
            "constructor": _constructor_from_policy_id(policy_id),
            "display_name": str(_field("display_name") or ""),
            "day": _field("day"),
            "sample_id": _field("sample_id"),
        }
    )


def _record_live_params(kind: str, payload: Dict[str, Any]) -> None:
    """Append one JSONL record when ``WSR_CAPTURE_STRATEGY_JSON`` is set.

    Used by the live ``main.py test_sim`` check so captured MS / RI / AC
    parameters can be compared to yaml without constructing a policy from a
    plain dict.

    Args:
        kind: ``packing``, ``route_improvement``, or ``acceptance_criteria``.
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


class _DictContext:
    """Attribute view over a test dict. Writes go back to the original mapping."""

    def __init__(self, data: Dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def as_kwargs(self) -> Dict[str, Any]:
        """Copy of the underlying mapping."""
        return dict(self._data)


def bind_context(context: Any) -> Any:
    """Return a context that supports attribute access.

    ``SimulationDayContext`` is returned as-is. A plain dict, which unit tests
    still pass, is wrapped so assignments update that dict.
    """
    if isinstance(context, _DictContext) or type(context).__name__ == "SimulationDayContext":
        return context
    if isinstance(context, dict):
        return _DictContext(context)
    return context


class SimulationAction(ABC):
    """
    Abstract base class for simulation actions.

    Defines the interface for all simulation commands. Each action receives
    a shared context dictionary and modifies it in-place with its outputs.

    Attributes:
        None
    """

    @abstractmethod
    def execute(self, context: Any) -> None:
        """
        Executes the action and updates the context in-place.

        Args:
            context: Shared dictionary containing simulation state.
        """
        pass
