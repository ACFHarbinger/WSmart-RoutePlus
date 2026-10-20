"""Pre-launch cost estimate of a job (CR-10).

Each task is assumed to run alone on one instance of the job's machine type for its
predicted wall time (CR-11), so a cell costs ``wall / 3600 x hourly price`` and the job
the sum over its cells; p10, p50 and p90 give the range. Cells without a predicted
time or a job without a known price give no estimate for that part (``None``), never zero.

Attributes:
    CostEstimate: A job's estimate.
    estimate_job: Build it from predictions and an hourly price.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class CostEstimate:
    """Estimated USD of a job.

    Attributes:
        low, mid, high: Job totals from the p10, p50 and p90 wall times (None when unknown).
        price_per_hour: USD per instance-hour used (None when unknown).
        cells: Per-cell ``{cell_id, low, mid, high}``.
        unpredicted_cells: Cells with no predicted wall time (excluded from the totals).
        assumptions: What the numbers do not include.
    """

    low: Optional[float]
    mid: Optional[float]
    high: Optional[float]
    price_per_hour: Optional[float]
    spot: bool
    machine_type: Optional[str]
    cells: List[Dict[str, Any]] = field(default_factory=list)
    unpredicted_cells: List[str] = field(default_factory=list)
    assumptions: str = (
        "one task per instance at list price for its predicted wall time; excludes storage, egress, "
        "billing minimums, discounts and spot interruptions"
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_job(
    predictions: Sequence[Any], price_per_hour: Optional[float], spot: bool, machine_type: Optional[str]
) -> CostEstimate:
    """The estimate for CR-11 ``predictions`` at ``price_per_hour``."""
    cells, missing = [], []
    for p in predictions:
        if p.p50 is None:
            missing.append(p.cell_id)
            continue
        cost = {
            q: None if price_per_hour is None else round(getattr(p, q) / 3600 * price_per_hour, 6)
            for q in ("p10", "p50", "p90")
        }
        cells.append({"cell_id": p.cell_id, "low": cost["p10"], "mid": cost["p50"], "high": cost["p90"]})

    def total(key: str) -> Optional[float]:
        if price_per_hour is None or not cells:
            return None
        return round(sum(c[key] for c in cells), 4)

    return CostEstimate(total("low"), total("mid"), total("high"), price_per_hour, spot, machine_type, cells, missing)
