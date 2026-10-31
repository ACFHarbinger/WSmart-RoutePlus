"""Campaign policy overlays: packing, improver and routing limit."""

import pytest
import yaml
from logic.src.pipeline.simulations.campaign_policies import (
    CAMPAIGN_PACKING,
    IMPROVERS,
    ROUTING_POLICIES,
    patch_policy,
    write_overlay,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_TREE = {
    "swc_tcf": {
        "gurobi": {
            "no_solution_time_factor": 9.0,
            "time_limit": 600.0,
            "packing": {"other/packing/pk_last_minute.yaml": ["last_minute_cf70"]},
            "route_improvement": {"other/route_improvement/ri_cls.yaml": ["default"]},
        },
        "other": {"time_limit": 30, "packing": {}},
    }
}


def test_packing_improver_and_limits_are_patched():
    out = patch_policy(_TREE, "ftsp")["swc_tcf"]
    assert out["gurobi"]["packing"] == CAMPAIGN_PACKING
    assert out["gurobi"]["route_improvement"] == IMPROVERS["ftsp"]
    assert out["gurobi"]["time_limit"] == 60.0
    assert out["other"]["time_limit"] == 30.0  # a tighter limit is kept
    assert out["gurobi"]["no_solution_time_factor"] == 1.0


def test_patching_does_not_modify_its_input():
    patch_policy(_TREE, "cls")
    assert _TREE["swc_tcf"]["gurobi"]["time_limit"] == 600.0


def test_unknown_improver_is_rejected():
    with pytest.raises(ValueError, match="improver"):
        patch_policy(_TREE, "two_opt")


@pytest.mark.parametrize("policy", ROUTING_POLICIES)
def test_every_campaign_policy_has_a_patched_overlay(policy, tmp_path):
    written = write_overlay(policy, "cls", tmp_path)
    data = yaml.safe_load(written.read_text())
    assert written.name.startswith("policy_")
    assert yaml.dump(data).count("last_minute_cf90") >= 1
    assert "last_minute_cf70" not in yaml.dump(data)


def test_strict_overlay_switches_the_fallbacks_off():
    from logic.src.pipeline.simulations.campaign_policies import patch_policy

    tree = {"bpc": {"custom": {"time_limit": 600.0, "packing": {}, "route_improvement": {}}}}
    assert patch_policy(tree, "cls", strict=True)["bpc"]["custom"]["allow_fallback"] is False
    assert "allow_fallback" not in patch_policy(tree, "cls")["bpc"]["custom"]


def test_strict_overlay_selects_the_paper_formulation():
    from logic.src.pipeline.simulations.campaign_policies import patch_policy

    tree = {"swc_tcf": {"gurobi": {"formulation": "directed", "time_limit": 600.0}}}
    assert patch_policy(tree, "cls", strict=True)["swc_tcf"]["gurobi"]["formulation"] == "asymmetric"
    assert patch_policy(tree, "cls")["swc_tcf"]["gurobi"]["formulation"] == "directed"
