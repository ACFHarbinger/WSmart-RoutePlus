"""Sample aggregation shared by the finishing state and the test orchestrator.

Failed samples keep their partial metrics and stay out of the mean. An
all-failed cell has no numeric estimate.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from logic.src.envs.common.metrics import (
    DEFAULT_METRICS,
    METRICS,
    MetricSummary,
    SampleResult,
    summarize_samples,
)


def coerce_metrics(payload: Any) -> Dict[str, float]:
    """Read canonical KPI numbers from a dict or a legacy positional vector."""
    from logic.src.constants import SIM_METRICS

    if isinstance(payload, Mapping):
        out: Dict[str, float] = {}
        for key, value in payload.items():
            if key in METRICS and isinstance(value, (int, float)) and not isinstance(value, bool):
                out[str(key)] = float(value)
        return out
    if isinstance(payload, (list, tuple)):
        return {
            key: float(value)
            for key, value in zip(SIM_METRICS, payload, strict=False)
            if key in METRICS and isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    raise TypeError(f"Cannot read metrics from {type(payload).__name__}")


def sample_from_payload(payload: Any, sample_id: str) -> SampleResult:
    """Build one sample record. A mapping may flag itself ``status: failed``."""
    status = "completed"
    failed_day: Optional[int] = None
    exception_type: Optional[str] = None
    if isinstance(payload, Mapping):
        if payload.get("status") == "failed":
            status = "failed"
        if payload.get("failed_day") is not None:
            failed_day = int(payload["failed_day"])
        if payload.get("exception_type"):
            exception_type = str(payload["exception_type"])
        if payload.get("sample_id") is not None:
            sample_id = str(payload["sample_id"])
    return SampleResult(
        sample_id=sample_id,
        metrics=coerce_metrics(payload),
        status=status,
        failed_day=failed_day,
        exception_type=exception_type,
    )


def summarize_payloads(
    payloads: Iterable[Any],
    names: Sequence[str] = DEFAULT_METRICS,
) -> MetricSummary:
    """Equal-weight summary. Failed payloads count in ``n_failed`` only."""
    samples: List[SampleResult] = []
    for index, payload in enumerate(payloads):
        samples.append(sample_from_payload(payload, str(index)))
    return summarize_samples(samples, names)


def summary_pair(
    payloads: Iterable[Any],
    names: Sequence[str] = DEFAULT_METRICS,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], MetricSummary]:
    """Mean and sample std dicts plus the full summary (includes n_failed)."""
    summary = summarize_payloads(payloads, names)
    return dict(summary.mean), dict(summary.std), summary
