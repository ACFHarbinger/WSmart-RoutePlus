"""Unit tests verifying Gemini review fixes across Lane C (Cloud Money & Scheduling)."""

import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from logic.controllers.cloud.cost.billing import import_billing_csv
from logic.controllers.cloud.cost.budget import BudgetExceeded, BudgetPolicy, CostMixin
from logic.controllers.cloud.model import JobSpec, Resources
from logic.controllers.cloud.prediction.features import _policy
from logic.controllers.cloud.prediction.history import get_base_predictor

pytestmark = [pytest.mark.unit]


def test_billing_csv_non_usd_currency_rejected(tmp_path: Path):
    """Verify import_billing_csv raises ValueError when currency is not USD."""
    csv_file = tmp_path / "billing.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsr_job_id", "cost_usd", "currency"])
        writer.writerow(["job-123", "10.50", "EUR"])

    cp = MagicMock()
    with pytest.raises(ValueError, match="non-USD billing currency 'EUR' is not supported"):
        import_billing_csv(cp, csv_file, provider="azure", at=1000.0)


def test_billing_csv_usd_accepted(tmp_path: Path):
    """Verify import_billing_csv accepts USD currency without error."""
    csv_file = tmp_path / "billing.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsr_job_id", "cost_usd", "currency"])
        writer.writerow(["job-123", "10.50", "USD"])

    cp = MagicMock()
    cp.list_jobs.return_value = [MagicMock(job_id="job-123")]
    res = import_billing_csv(cp, csv_file, provider="azure", at=1000.0)
    assert res == {"job-123": 10.50}


def test_feature_extraction_policy_amgat_mapped_to_am():
    """Verify _policy maps 'na.amgat', 'amgat', and 'na' to 'am'."""
    assert _policy({"sim.policies": "na.amgat"}, []) == "am"
    assert _policy({"sim.policies": "amgat"}, []) == "am"
    assert _policy({"sim.policies": "na"}, []) == "am"
    assert _policy({"sim.policies": "[na.amgat]"}, []) == "am"


def test_feature_extraction_unbracketed_policy():
    """Verify _policy extracts unbracketed policy names."""
    assert _policy({"sim.policies": "alns"}, []) == "alns"
    assert _policy({"policy": "gurobi"}, []) == "gurobi"


def test_get_base_predictor_cached():
    """Verify get_base_predictor caches and returns the same static instance."""
    pred1 = get_base_predictor()
    pred2 = get_base_predictor()
    assert pred1 is pred2


def test_budget_monthly_cap_exemption_for_local_jobs():
    """Verify local jobs are permitted even when monthly budget is exhausted."""
    mixin = CostMixin()
    mixin.budget = BudgetPolicy(monthly_usd=100.0)
    mixin.clock = lambda: 1700000000.0
    mixin.month_committed = lambda now: 150.0  # Already over budget

    local_spec = JobSpec(
        name="local-test",
        provider="local",
        compute="process",
        cells=[],
        resources=Resources(budget_usd=None),
    )

    # Should not raise BudgetExceeded for local job without estimate
    warnings = mixin._check_budget(local_spec, estimate=None)
    assert warnings == []


def test_budget_monthly_cap_enforced_for_remote_jobs_without_estimate():
    """Verify remote jobs without estimate raise BudgetExceeded when monthly budget is exhausted."""
    mixin = CostMixin()
    mixin.budget = BudgetPolicy(monthly_usd=100.0)
    mixin.clock = lambda: 1700000000.0
    mixin.month_committed = lambda now: 150.0  # Over budget

    remote_spec = JobSpec(
        name="remote-test",
        provider="azure",
        compute="container",
        cells=[],
        resources=Resources(budget_usd=None),
    )

    with pytest.raises(BudgetExceeded, match="monthly budget exceeded"):
        mixin._check_budget(remote_spec, estimate=None)


def test_month_committed_memoization():
    """Verify month_committed retrieves from _month_memo when available."""
    mixin = CostMixin()
    mixin._month_memo = {"committed_2023-11": 42.123456}
    committed = mixin.month_committed(1700000000.0)  # Nov 2023
    assert committed == 42.123456
