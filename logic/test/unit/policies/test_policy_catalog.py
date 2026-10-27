"""The module catalog includes every registered id and reports import failures."""

import pytest
from logic.src.policies.acceptance_criteria._manifest import ENTRIES as ACCEPTANCE_ENTRIES
from logic.src.policies.catalog import build_catalog, catalog_ids, missing_manifest_ids
from logic.src.policies.packing._manifest import ENTRIES as PACKING_ENTRIES
from logic.src.policies.route_improvement._manifest import ENTRIES as IMPROVEMENT_ENTRIES
from logic.src.policies.routing._manifest import ENTRIES as ROUTING_ENTRIES
from logic.src.utils.registry import PACKING

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_manifest_ids_are_in_the_catalog():
    catalog = build_catalog(resolve=False)
    assert missing_manifest_ids(catalog) == []
    assert set(PACKING_ENTRIES) <= set(catalog_ids(catalog, "packing"))
    assert set(ROUTING_ENTRIES) <= set(catalog_ids(catalog, "routing"))
    assert set(IMPROVEMENT_ENTRIES) <= set(catalog_ids(catalog, "route_improvement"))
    assert set(ACCEPTANCE_ENTRIES) <= set(catalog_ids(catalog, "acceptance"))
    assert "destroy_ruin" in catalog_ids(catalog, "operators")
    assert "capacity_separation" in catalog_ids(catalog, "separation")
    models = {row["id"]: row for row in catalog["models"]}
    assert models["am"]["status"] == "shipped"
    assert models["WeightAdjustmentRNN"]["role"] == "reward_weight_adjustment"


def test_live_registry_declaration_appears_without_a_studio_edit():
    PACKING.declare("fixture_pack_catalog", "logic.src.policies.packing.last_minute:LastMinuteSelection")
    try:
        catalog = build_catalog(resolve=False)
        assert "fixture_pack_catalog" in catalog_ids(catalog, "packing")
    finally:
        PACKING._specs.pop("fixture_pack_catalog", None)


def test_failed_import_is_an_entry_not_a_crash():
    PACKING.declare("fixture_pack_missing", "logic.src.policies.packing.does_not_exist:Missing")
    try:
        catalog = build_catalog(resolve=True, resolve_ids={"fixture_pack_missing"})
        row = next(item for item in catalog["packing"] if item["id"] == "fixture_pack_missing")
        assert row["error"]
        assert "does_not_exist" in row["error"] or "ModuleNotFoundError" in row["error"]
    finally:
        PACKING._specs.pop("fixture_pack_missing", None)
