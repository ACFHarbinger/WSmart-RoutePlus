"""The per-policy RNG seed depends on the constructor only (B-grok-01)."""

import pytest
from logic.src.pipeline.simulations.day_context import get_canonical_policy_name

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("last_minute_cf70_aco_hh_custom_cls", "aco_hh"),
        ("last_minute_cf90_pg_clns_custom_cls", "pg_clns"),
        ("last_minute_cf70_psoma_bmc_ftsp", "psoma"),
        ("service_level2_sans_new_ftsp", "sans"),
        ("lookahead_swc_tcf_gurobi", "swc_tcf"),
        ("lookahead_na_amgat_emp", "na"),
        ("lookahead_ma_ts_custom_gamma3", "ma_ts"),
    ],
)
def test_constructor_key_is_extracted(slug, expected):
    assert get_canonical_policy_name(slug) == expected


def test_selector_and_threshold_do_not_change_the_seed_key():
    assert get_canonical_policy_name("last_minute_cf70_alns_custom_bmc_cls") == get_canonical_policy_name(
        "lookahead_alns_custom_bmc_ftsp"
    )
    assert get_canonical_policy_name("last_minute_cf70_alns") != get_canonical_policy_name("last_minute_cf70_hgs")
