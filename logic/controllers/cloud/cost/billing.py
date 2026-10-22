"""Import of provider billing exports (CR-10).

Backends label what they create with the job id (``wsr_job_id``), so a billing export
filtered to that label attributes charges to jobs. The importer reads a CSV export
(GCP BigQuery billing export, AWS Cost and Usage Report or Azure cost export, saved
as CSV) and records one ``billing`` event per job with its invoiced total, next to the
accrued estimate. Column names differ per provider; :data:`COLUMNS` lists the ones
recognised for the job label and the cost.

Attributes:
    COLUMNS: Recognised column names.
    import_billing_csv: Record a billing export's per-job totals; returns them.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Optional

from logic.controllers.cloud.interfaces import ControlPlane
from logic.controllers.cloud.model import Event

COLUMNS = {
    "job": ("wsr_job_id", "labels.wsr_job_id", "resourceTags/user:wsr_job_id", "tags/wsr_job_id", "job_id"),
    "cost": ("cost_usd", "cost", "lineItem/UnblendedCost", "costInBillingCurrency", "CostInBillingCurrency"),
    "currency": ("billingCurrency", "BillingCurrency", "currency", "CurrencyCode", "currency_code"),
}


# "1,234.56": comma thousands separators with a point decimal, as the USD exports write them.
_THOUSANDS = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")

def _column(header, names) -> Optional[str]:
    lowered = {h.lower(): h for h in header}
    return next((lowered[n.lower()] for n in names if n.lower() in lowered), None)


def import_billing_csv(cp: ControlPlane, path: Path, provider: str, at: float) -> Dict[str, float]:
    """Sum ``path``'s charges per job label and record a ``billing`` event on each known job.

    Rows without a job label, and labels that are not jobs of this control plane, are ignored.
    Re-importing the same export appends a new event; readers use the latest.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        job_col = _column(reader.fieldnames or [], COLUMNS["job"])
        cost_col = _column(reader.fieldnames or [], COLUMNS["cost"])
        currency_col = _column(reader.fieldnames or [], COLUMNS["currency"])
        if job_col is None or cost_col is None:
            raise ValueError(f"{path}: no job label column ({COLUMNS['job']}) or cost column ({COLUMNS['cost']})")
        totals: Dict[str, float] = {}
        for row in reader:
            if currency_col:
                curr = (row.get(currency_col) or "USD").strip().upper()
                if curr and curr != "USD":
                    raise ValueError(f"{path}: non-USD billing currency '{curr}' is not supported")
            job_id = (row.get(job_col) or "").strip()
            if not job_id:
                continue
            raw = (row.get(cost_col) or "").strip()
            if _THOUSANDS.match(raw):
                raw = raw.replace(",", "")
            try:
                cost = float(raw) if raw else 0.0
            except ValueError:
                # Skipping the row would under-report the job's bill.
                raise ValueError(f"{path}: unparseable cost {row.get(cost_col)!r} for job '{job_id}'") from None
            totals[job_id] = totals.get(job_id, 0.0) + cost
    known = {job.job_id for job in cp.list_jobs(limit=10_000)}
    recorded = {}
    for job_id, total in sorted(totals.items()):
        if job_id in known:
            cp.append_event(
                Event(
                    job_id=job_id,
                    kind="billing",
                    data={"provider": provider, "cost_usd": round(total, 6), "source": str(path)},
                    at=at,
                )
            )
            recorded[job_id] = round(total, 6)
    return recorded
