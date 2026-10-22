"""_check_budget without an estimate reads the month's committed spend once."""

import pytest

from logic.controllers.cloud.cost import BudgetExceeded, BudgetPolicy

from logic.test.unit.controllers.cloud.test_cost import _service, _spec

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize(("committed", "refused"), [(50.0, False), (100.0, True), (150.0, True)])
def test_no_estimate_month_check_reads_committed_spend_once(tmp_path, monkeypatch, committed, refused):
    service, _ = _service(tmp_path, budget=BudgetPolicy(monthly_usd=100.0))
    calls = []

    def month_committed(now):
        calls.append(now)
        return committed

    monkeypatch.setattr(service, "month_committed", month_committed)
    if refused:
        with pytest.raises(BudgetExceeded, match=f"{100.0 - committed:.2f} USD left"):
            service._check_budget(_spec(), None)
    else:
        assert service._check_budget(_spec(), None) == []
    assert len(calls) == 1
