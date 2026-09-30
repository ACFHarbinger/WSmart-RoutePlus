"""Every ``test_sim`` policy entry must resolve, register and build (issue #61).

The 2026-09-29 round found three broken entries: ``gp_hh`` was registered
under a key nothing else used ("gphh"), ``mp_bmc`` had no implementation at
all, and ``src`` crashed in ``_build_config`` because its
``_get_config_key`` was a classmethod while the base calls
``cls._get_config_key(cls)``.

This test walks the default ``sim.policies`` list of
``logic/configs/tasks/test_sim.yaml`` — exactly what the simulator fans
out — and for every entry does what the live path does
(``actions/route_construction.py``): compose the task the way ``main.py``
does, resolve the entry's yaml node, find a registered constructor through
the same solver-key chain (explicit ``policy.type``/``solver``/``engine``
key, registry-key match, then the compound-name word-boundary fallback
that resolves variants like ``sans_new`` → ``sans``), and build the
adapter with the section passed as a raw dict with the simulator's seed
injection. The adapter's typed config must equal an independent
``_build_config`` of the same section, so nothing between ``get_adapter``
and the dataclass drops or mutates the yaml values.

Before the fixes this failed on ``gp_hh`` (no registered constructor for
the entry name, compound fallback included), ``mp_bmc`` (no
implementation anywhere) and ``src`` (TypeError: ``_get_config_key()``
takes 1 positional argument but 2 were given).
"""

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from hydra import compose, initialize_config_dir
from logic.src.policies.route_construction.base.factory import RouteConstructorFactory
from logic.src.policies.route_construction.base.registry import RouteConstructorRegistry
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIGS_DIR = _REPO_ROOT / "logic" / "configs"
_TEST_SIM = _CONFIGS_DIR / "tasks" / "test_sim.yaml"

# The seed the simulator injects into every policy's raw config
# (actions/route_construction.py, run_day's policy_seed fallback).
_SIM_SEED = 1234

_CFG = None


def _compose_test_sim() -> Any:
    """Compose the task once per session, exactly as main.py does.

    The root config plus the ``tasks=test_sim`` override — the task file's
    ``override /tracking`` only resolves inside that defaults tree.
    """
    global _CFG
    if _CFG is None:
        with initialize_config_dir(str(_CONFIGS_DIR), version_base=None):
            _CFG = compose(config_name="config", overrides=["tasks=test_sim"])
        RouteConstructorFactory.ensure_registered()
    return _CFG


def _policy_keys() -> List[str]:
    """The keys of the default ``sim.policies`` list, read statically so
    pytest can parametrize without composing first."""
    text = _TEST_SIM.read_text(encoding="utf-8")
    in_policies = False
    keys = []
    for line in text.splitlines():
        if re.match(r"^sim:", line):
            in_policies = True
            continue
        if in_policies:
            m = re.match(r"^    - ([A-Za-z0-9_]+):", line)
            if m:
                keys.append(m.group(1))
            elif line and not line.startswith(" "):
                break  # left the sim: block (blank lines are fine)
    assert keys, "no sim.policies entries found in test_sim.yaml"
    return keys


def _raw_entry_config(cfg: Any, key: str) -> Dict[str, Any]:
    """Resolve the entry's yaml node the way the simulator does."""
    for entry in cfg.sim.policies:
        if key in entry:
            raw = {key: OmegaConf.to_container(entry[key], resolve=True)}
            raw["seed"] = _SIM_SEED
            return raw
    raise AssertionError(f"entry {key!r} not found in composed sim.policies")


def _resolve_solver_key(raw: Dict[str, Any], key: str) -> Optional[str]:
    """The action's solver-key chain, minus the display-name last resort
    (which only exists on the live per-day path)."""
    registered = set(RouteConstructorRegistry.list_route_constructors())
    # 1. Explicit engine/solver/type inside the policy section.
    for path in ("policy.type", "policy.solver", "policy.engine"):
        node = raw.get(key)
        if isinstance(node, dict):
            sub = node.get("policy", {})
            if isinstance(sub, dict) and sub.get(path.split(".", 1)[1]):
                return str(sub[path.split(".", 1)[1]]).lower()
    # 2. A registered key appearing as a key of the raw config.
    if key in registered:
        return key
    # 3. Compound-name fallback (e.g. 'sans_new' -> 'sans').
    lower = key.lower()
    for eng in sorted(registered, key=len, reverse=True):
        if f"_{eng}_" in lower or lower.startswith(f"{eng}_") or lower.endswith(f"_{eng}") or eng == lower:
            return eng
    return None


@pytest.mark.unit
@pytest.mark.parametrize("key", _policy_keys())
def test_test_sim_entry_resolves_and_builds(key: str):
    cfg = _compose_test_sim()
    raw = _raw_entry_config(cfg, key)

    # 1. The entry maps to a registered route constructor through the same
    #    chain the live action uses.
    solver_key = _resolve_solver_key(raw, key)
    assert solver_key is not None, (
        f"test_sim entry {key!r} resolves to no registered route constructor "
        "(explicit key, registry key and compound-name fallback all failed); "
        "fix the registry key or remove the entry."
    )

    # 2. The adapter builds from the section, as the simulator calls it.
    adapter = RouteConstructorFactory.get_adapter(solver_key, config=raw)

    # 3. The adapter's typed config equals an independent build from the
    #    same section (policies with their own config parsing, such as the
    #    joint policies, are exempt from the BaseRoutingPolicy contract).
    cls = RouteConstructorRegistry.get(solver_key)
    if cls._config_class() is not None and hasattr(cls, "_build_config"):
        expected, _seed = cls._build_config(OmegaConf.create(raw))
        assert isinstance(adapter.config, cls._config_class()), key
        assert asdict(adapter.config) == asdict(expected), key


@pytest.mark.unit
def test_policies_references_point_at_defaults_sections():
    """Every ``sim.policies`` reference interpolates a policy-group section
    that the defaults list actually provides (``${p.<section>...}``), so no
    entry can dangle after a rename or removal (the mp_bmc class of bug)."""
    text = _TEST_SIM.read_text(encoding="utf-8")
    defaults = set(re.findall(r"^\s*- /policies@p\.([A-Za-z0-9_]+):", text, re.M))
    refs = re.findall(r"^\s*- ([A-Za-z0-9_]+): \$\{p\.([A-Za-z0-9_]+)\.", text, re.M)
    assert refs, "no sim.policies interpolations found"
    for _key, section in refs:
        assert section in defaults, (
            f"sim.policies references policy section {section!r} that the defaults list never provides"
        )
