"""Every registered route constructor must honour the config contract (issue #61, rounds 3+5).

The 2026-09-29 round fixed the three broken ``test_sim`` entries but only
covered the entries ``test_sim`` actually fans out to. ``arco`` is registered
and has a yaml, yet had no ``test_sim`` entry, so its defects were invisible:
``_get_config_key`` was a ``@classmethod`` while
``BaseRoutingPolicy._build_config`` calls ``cls._get_config_key(cls)``
(the same crash ``src`` had), and its constructor pool defaulted to ``"nn"``,
a constructor name that has never been registered (no commit ever added it;
the registry holds no nearest-neighbour constructor).

This test generalises the check to the whole registry, not just ``test_sim``:

1. For **every** registered route constructor, ``_get_config_key`` must be
   callable exactly the way ``base_routing_policy.py`` calls it and return a
   non-empty string. A ``@classmethod`` override raises ``TypeError`` there,
   and a missing override raises ``AttributeError``.
2. Every constructor name in a default config or yaml must exist in the
   registry — the typed config defaults (``logic/src/configs/policies``), the
   runtime params defaults and ``from_config`` fallbacks
   (``route_construction/**/params.py``) and every ``constructors`` pool in
   ``logic/configs/policies/policy_*.yaml``. A pool naming an unregistered
   constructor can only fail at run time with
   ``ValueError: Unknown policy`` from ``RouteConstructorFactory.get_adapter``.
3. Every **orchestrator** — a registered policy whose yaml fans out to a
   ``constructors`` pool — must be exercised by a ``test_sim`` entry (an
   orchestrator's adapter is only built through one, which is exactly why
   ``arco`` stayed invisible) and must set a ``mandatory_selection``
   (without one, every day's mandatory set is empty and the orchestrator
   collects nothing — the ``src`` lesson).

Before the round-3 fixes this failed on ``arco`` (TypeError from the
classmethod) and on the ``"nn"`` pool names in ``ARCOParams``, ``ARCOConfig``
and ``policy_arco.yaml``. Before the round-5 entry, contract 3 failed on
``arco`` (no ``test_sim`` entry).
"""

import importlib
import pkgutil
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Tuple

import pytest
import yaml
from logic.src.policies.route_construction.base.factory import RouteConstructorFactory
from logic.src.policies.route_construction.base.registry import RouteConstructorRegistry

_REPO_ROOT = Path(__file__).resolve().parents[5]
_POLICY_YAMLS = sorted((_REPO_ROOT / "logic" / "configs" / "policies").glob("policy_*.yaml"))
_TEST_SIM = _REPO_ROOT / "logic" / "configs" / "tasks" / "test_sim.yaml"

RouteConstructorFactory.ensure_registered()
_REGISTERED = sorted(RouteConstructorRegistry.list_route_constructors())


def _is_dataclass_with_constructors(obj: Any) -> bool:
    """True for a dataclass type declaring a ``constructors`` pool field."""
    return isinstance(obj, type) and is_dataclass(obj) and "constructors" in [f.name for f in fields(obj)]


def _params_pools() -> List[Tuple[str, List[str]]]:
    """Constructor pools from every ``route_construction/**/params.py`` dataclass.

    Covers both the dataclass defaults and the ``from_config`` fallback used
    when the config object lacks the field.
    """
    import logic.src.policies.route_construction as route_construction

    pools: List[Tuple[str, List[str]]] = []
    for info in pkgutil.walk_packages(route_construction.__path__, route_construction.__name__ + "."):
        if info.name.rsplit(".", 1)[-1] != "params":
            continue
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if not _is_dataclass_with_constructors(obj):
                continue
            try:
                instance = obj()
            except TypeError:
                pass  # Required-field dataclasses are covered through from_config below.
            else:
                pools.append((f"{obj.__name__} default", list(instance.constructors)))
            if callable(getattr(obj, "from_config", None)):
                fallback = obj.from_config(SimpleNamespace())
                pools.append((f"{obj.__name__} from_config fallback", list(fallback.constructors)))
    return pools


def _config_pools() -> List[Tuple[str, List[str]]]:
    """Constructor pools from the typed config defaults in ``configs/policies``."""
    import logic.src.configs.policies as config_policies

    pools: List[Tuple[str, List[str]]] = []
    for info in pkgutil.walk_packages(config_policies.__path__, config_policies.__name__ + "."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if not _is_dataclass_with_constructors(obj):
                continue
            pools.append((f"{obj.__name__} default", list(obj().constructors)))
    return pools


def _yaml_pools() -> List[Tuple[str, List[str]]]:
    """Constructor pools from every ``policy_*.yaml`` defaults file."""

    def walk(node: Any, source: str, path: List[str]) -> Iterator[Tuple[str, List[str]]]:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "constructors" and isinstance(value, list):
                    yield (f"{source} {'.'.join(path + [str(key)])}", [str(v) for v in value])
                yield from walk(value, source, path + [str(key)])
        elif isinstance(node, list):
            for i, value in enumerate(node):
                yield from walk(value, source, path + [str(i)])

    pools: List[Tuple[str, List[str]]] = []
    for yaml_file in _POLICY_YAMLS:
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        relative = str(yaml_file.relative_to(_REPO_ROOT))
        pools.extend(walk(data, relative, []))
    return pools


def _all_pool_rows() -> List[Tuple[str, str]]:
    """(source, constructor name) rows: one row per name in every pool."""
    rows: List[Tuple[str, str]] = []
    for source, names in _params_pools() + _config_pools() + _yaml_pools():
        for name in names:
            rows.append((source, name))
    assert rows, "no constructor pools found in params, configs or policy yamls"
    return rows


@pytest.mark.unit
@pytest.mark.parametrize("name", _REGISTERED)
def test_config_key_is_callable_the_base_way(name: str):
    """``_build_config`` calls ``cls._get_config_key(cls)`` on every adapter."""
    cls = RouteConstructorRegistry.get(name)
    key = cls._get_config_key(cls)  # the exact base_routing_policy.py call
    assert isinstance(key, str) and key, f"{name}: _get_config_key must return a non-empty string"


@pytest.mark.unit
@pytest.mark.parametrize("source,name", _all_pool_rows(), ids=[f"{s}::{n}" for s, n in _all_pool_rows()])
def test_constructor_pool_names_are_registered(source: str, name: str):
    """A pool name the factory cannot resolve can only fail at run time."""
    assert name in set(_REGISTERED), (
        f"{source} names constructor {name!r}, which is not registered in "
        "RouteConstructorRegistry; RouteConstructorFactory.get_adapter would "
        "raise ValueError: Unknown policy."
    )


# ---------------------------------------------------------------------------
# Contract 3 (round 5): orchestrators must be exercised by test_sim and must
# select a mandatory set, or they are invisible and collect nothing.
# ---------------------------------------------------------------------------


def _orchestrators() -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """(registry key -> (yaml file, merged yaml section)) for every registered
    policy whose yaml fans out to a ``constructors`` pool."""
    registered = set(_REGISTERED)
    found: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for yaml_file in _POLICY_YAMLS:
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        for key, section in data.items():
            if key not in registered or not isinstance(section, list):
                continue
            merged = {k: v for item in section if isinstance(item, dict) for k, v in item.items()}
            if isinstance(merged.get("constructors"), list):
                found[key] = (str(yaml_file.relative_to(_REPO_ROOT)), merged)
    assert found, "no orchestrator (constructor-pool) policy yamls found"
    return found


def _test_sim_policy_keys() -> List[str]:
    """The keys of the default ``sim.policies`` list, read statically so no
    Hydra composition is needed."""
    in_policies = False
    keys: List[str] = []
    for line in _TEST_SIM.read_text(encoding="utf-8").splitlines():
        if re.match(r"^sim:", line):
            in_policies = True
            continue
        if in_policies:
            m = re.match(r"^    - ([A-Za-z0-9_]+):", line)
            if m:
                keys.append(m.group(1))
            elif line and not line.startswith(" "):
                break  # left the sim: block
    assert keys, "no sim.policies entries found in test_sim.yaml"
    return keys


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(_orchestrators()))
def test_orchestrator_has_a_test_sim_entry(key: str):
    """An orchestrator's adapter is only built through a ``test_sim`` entry;
    without one neither the simulator nor the entries test ever builds it."""
    yaml_file, _ = _orchestrators()[key]
    assert key in _test_sim_policy_keys(), (
        f"{key} is a registered orchestrator ({yaml_file} fans out to a "
        "constructors pool) but has no test_sim entry; its adapter and "
        "sub-constructor wiring are never exercised."
    )


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(_orchestrators()))
def test_orchestrator_yaml_sets_a_mandatory_selection(key: str):
    """Without a mandatory_selection every day's mandatory set is empty and
    the orchestrator collects nothing (the ``src`` lesson)."""
    yaml_file, merged = _orchestrators()[key]
    assert merged.get("mandatory_selection"), (
        f"{yaml_file} defines an orchestrator but sets no mandatory_selection; "
        f"{key} would collect 0 kg in every simulated day."
    )
