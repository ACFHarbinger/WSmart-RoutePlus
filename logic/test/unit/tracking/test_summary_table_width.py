"""Summary tables keep their metric columns when stdout is not a terminal."""

import pytest

from logic.src.tracking.logging.modules.summary_display import display_simulation_summary_table

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_redirected_summary_shows_every_metric(capsys) -> None:
    policy = "last_minute_cf70_alns_custom_calibrated_bmc_cls"
    display_simulation_summary_table({policy: {"profit": 345.66, "km": 536.85, "kg": 1511.92, "overflows": 6}})
    out = capsys.readouterr().out
    assert policy in out
    assert "$345.66" in out and "536.9km" in out and "1,511.9kg" in out
