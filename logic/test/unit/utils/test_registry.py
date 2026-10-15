"""Unified Registry (P-01) and lazy classical imports (P-02)."""

import subprocess
import sys
from textwrap import dedent

import pytest

from logic.src.utils.registry import Registry

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_registry_name_lookup_create_and_tags():
    stage = Registry("demo")

    class Foo:
        def __init__(self, value=1):
            self.value = value

    stage.register("foo", tags=("alpha",))(Foo)
    assert stage.get("FOO") is Foo
    assert "foo" in stage.names()
    inst = stage.create("foo", value=3)
    assert inst.value == 3
    assert stage.query_intersection("alpha") == [Foo]
    assert stage.query_union("alpha", "missing") == [Foo]
    assert stage.query_difference(["alpha"], ["beta"]) == [Foo]
    assert Foo in stage.get_all()


def test_parent_tag_union():
    root = Registry("all")
    left = Registry("left", parent=root)
    right = Registry("right", parent=root)

    class A:
        pass

    class B:
        pass

    left.register("a", tags=("shared", "left"))(A)
    right.register("b", tags=("shared", "right"))(B)
    found = root.query_intersection("shared")
    assert A in found and B in found


def test_packing_registry_declares_without_vectorized():
    from logic.src.policies.packing import PackingFactory, PackingRegistry

    assert PackingRegistry.get("last_minute") is not None
    assert PackingRegistry.get("revenue_threshold") is PackingRegistry.get("revenue")
    strategy = PackingFactory.create_strategy("last_minute")
    assert strategy is not None
    assert not any(name.startswith("logic.src.policies.packing.vectorized") for name in sys.modules)


def test_released_acceptance_and_improver_create():
    from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
    from logic.src.policies.route_improvement.base.factory import RouteImproverFactory

    bmc = AcceptanceCriterionFactory.create("bmc", initial_temp=1000.0, alpha=0.995)
    oi = AcceptanceCriterionFactory.create("oi")
    cls = RouteImproverFactory.create("cls")
    ftsp = RouteImproverFactory.create("ftsp")
    assert bmc is not None and oi is not None
    assert cls is not None and ftsp is not None


def test_routing_manifest_covers_declared_policy_modules():
    import re
    from pathlib import Path

    from logic.src.policies.routing.base.registry import RoutingAlgorithmRegistry

    root = Path(__file__).resolve().parents[3] / "src" / "policies" / "routing"
    declared = set()
    for path in root.rglob("policy_*.py"):
        declared |= set(re.findall(r'RoutingAlgorithmRegistry\.register\("([^"]+)"\)', path.read_text()))
    missing = declared - set(RoutingAlgorithmRegistry.list_routing_algorithms())
    assert not missing, f"declared but never registered: {sorted(missing)}"
    assert {"swc_tcf", "alns", "hgs"} <= declared


def test_hydra_vector_config_aliases():
    from logic.src.configs.rl.policies import (
        ALNSConfig,
        HGSALNSConfig,
        HGSConfig,
        HVPLConfig,
        ILSConfig,
        VectorALNSConfig,
        VectorHGSALNSConfig,
        VectorHGSConfig,
        VectorHVPLConfig,
        VectorILSConfig,
    )

    assert ALNSConfig is VectorALNSConfig
    assert HGSConfig is VectorHGSConfig
    assert ILSConfig is VectorILSConfig
    assert HGSALNSConfig is VectorHGSALNSConfig
    assert HVPLConfig is VectorHVPLConfig


def test_released_classical_imports_do_not_load_torch():
    script = dedent(
        """
        import sys
        from logic.src.policies.packing.base.factory import PackingFactory
        from logic.src.policies.acceptance_criteria.base.factory import AcceptanceCriterionFactory
        from logic.src.policies.route_improvement.base.factory import RouteImproverFactory
        import logic.src.models as models

        PackingFactory.create_strategy("last_minute")
        AcceptanceCriterionFactory.create("bmc", initial_temp=1000.0, alpha=0.995)
        AcceptanceCriterionFactory.create("oi")
        RouteImproverFactory.create("cls")
        RouteImproverFactory.create("ftsp")
        assert "AttentionModel" in models.__all__
        assert "logic.src.models.core.attention_model" not in sys.modules
        banned = [
            name
            for name in sys.modules
            if name == "torch"
            or name == "torchrl"
            or name == "tensordict"
            or name == "torch_geometric"
            or name.startswith(("torch.", "torchrl.", "tensordict.", "torch_geometric."))
        ]
        assert not banned, banned
        """
    )
    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
