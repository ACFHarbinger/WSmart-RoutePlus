"""Every declared route constructor is registered, whatever was imported before (C2, #79)."""

import re
from pathlib import Path

import pytest
from logic.src.policies.route_construction.base.factory import RouteConstructorFactory
from logic.src.policies.route_construction.base.registry import RouteConstructorRegistry

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[3] / "src" / "policies" / "route_construction"


def test_ensure_registered_loads_every_policy_module():
    declared = set()
    for path in ROOT.rglob("policy_*.py"):
        declared |= set(re.findall(r'RouteConstructorRegistry\.register\("([^"]+)"\)', path.read_text()))
    RouteConstructorFactory.ensure_registered()
    missing = declared - set(RouteConstructorRegistry.list_route_constructors())
    assert not missing, f"declared but never registered: {sorted(missing)}"
    assert {"swc_tcf", "egh", "lasm"} <= declared
