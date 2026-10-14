"""Every declared routing algorithm is registered, whatever was imported before (C2, #79)."""

import re
from pathlib import Path

import pytest
from logic.src.policies.routing.base.factory import RoutingAlgorithmFactory
from logic.src.policies.routing.base.registry import RoutingAlgorithmRegistry

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[3] / "src" / "policies" / "routing"


def test_ensure_registered_loads_every_policy_module():
    declared = set()
    for path in ROOT.rglob("policy_*.py"):
        declared |= set(re.findall(r'RoutingAlgorithmRegistry\.register\("([^"]+)"\)', path.read_text()))
    RoutingAlgorithmFactory.ensure_registered()
    missing = declared - set(RoutingAlgorithmRegistry.list_routing_algorithms())
    assert not missing, f"declared but never registered: {sorted(missing)}"
    assert {"swc_tcf", "egh", "lasm"} <= declared
