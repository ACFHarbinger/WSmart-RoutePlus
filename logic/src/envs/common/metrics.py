"""Torch-free KPI registry and explicit day/sample aggregation.

Day records use canonical metric keys. Route feasibility violations are NOT bin
``overflows``; route evaluators must keep those diagnostics separate. Failed
samples retain their partial records but never contribute to population summaries.
"""

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from .objectives import Economics, profit_objective


@dataclass(frozen=True)
class DayOutcome:
    """Physical quantities from the executed (post-repair) day's plan.

    ``time`` is policy compute seconds, ``day_wall_time`` is complete day wall
    seconds and ``time_spent`` is physical shift hours. Missing shift time stays
    absent rather than being silently counted as zero.
    """

    collected_kg: float
    distance_km: float
    overflows: int = 0
    collection_count: int = 0
    lost_kg: float = 0.0
    policy_time_s: float = 0.0
    day_wall_time_s: float = 0.0
    time_spent_h: Optional[float] = None
    collection_day: bool = False

    def metrics(self, economics: Economics) -> Dict[str, float]:
        """Map physical outcomes to canonical KPI keys using the shared objective."""
        result = {
            "kg": self.collected_kg,
            "km": self.distance_km,
            "overflows": float(self.overflows),
            "ncol": float(self.collection_count),
            "kg_lost": self.lost_kg,
            "profit": profit_objective(
                self.collected_kg, self.distance_km, revenue_kg=economics.revenue_kg, cost_km=economics.cost_km
            ),
            "time": self.policy_time_s,
            "day_wall_time": self.day_wall_time_s,
            "days_simulated": 1.0,
            "collection_days": float(self.collection_day),
        }
        if self.time_spent_h is not None:
            result["time_spent"] = self.time_spent_h
        return result


@dataclass(frozen=True)
class Metric:
    """KPI metadata and its within-sample aggregation rule.

    Ratio dependencies are summed over days before dividing. Across samples all
    metrics use equal-weight mean and sample standard deviation (ddof=1).
    """

    name: str
    unit: str
    direction: str
    label: str
    aggregate: str = "sum"
    numerator: Optional[str] = None
    denominator: Optional[str] = None
    per_day: Optional[Callable[[Mapping[str, float]], float]] = None

    def __post_init__(self) -> None:
        if self.direction not in ("min", "max", "none"):
            raise ValueError("Metric direction must be min, max or none")
        if self.aggregate not in ("sum", "mean", "ratio_of_sums"):
            raise ValueError("Unknown metric aggregation rule")
        if self.aggregate == "ratio_of_sums" and not (self.numerator and self.denominator):
            raise ValueError("A ratio metric requires numerator and denominator dependencies")


class MetricRegistry(Mapping[str, Metric]):
    """Named metric definitions; unknown, duplicate or cyclic definitions fail loudly."""

    def __init__(self, metrics: Iterable[Metric] = ()) -> None:
        self._metrics: Dict[str, Metric] = {}
        for metric in metrics:
            self.register(metric)

    def register(self, metric: Metric) -> None:
        if metric.name in self._metrics:
            raise ValueError(f"Duplicate metric: {metric.name}")
        if metric.aggregate == "ratio_of_sums":
            for dependency in (metric.numerator, metric.denominator):
                if dependency not in self._metrics:
                    raise ValueError(f"Register dependency {dependency!r} before {metric.name!r}")
                if self._metrics[dependency].aggregate != "sum":
                    raise ValueError("Ratio dependencies must be additive quantities")
        self._metrics[metric.name] = metric

    def __getitem__(self, name: str) -> Metric:
        return self._metrics[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def value(self, name: str, day: Mapping[str, float]) -> float:
        """Evaluate one day's KPI; absent physical quantities are errors."""
        metric = self[name]
        if metric.aggregate == "ratio_of_sums":
            numerator = self.value(metric.numerator, day)
            denominator = self.value(metric.denominator, day)
            value = numerator / denominator if denominator > 0 else 0.0
            if not math.isfinite(value):
                raise ValueError(f"Non-finite value for {name!r}")
            return value
        value = float(metric.per_day(day) if metric.per_day else day[name])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value for {name!r}")
        return value

    def aggregate_days(self, days: Iterable[Mapping[str, float]], names: Iterable[str]) -> Dict[str, float]:
        """Aggregate a sample, retaining ratio dependencies for population summaries.

        Empty horizons have zero totals; a mean of no observations is undefined.
        """
        records = tuple(days)
        selected = list(dict.fromkeys(names))
        for name in tuple(selected):
            metric = self[name]
            if metric.aggregate == "ratio_of_sums":
                for dependency in (metric.numerator, metric.denominator):
                    if dependency not in selected:
                        selected.append(dependency)
        result: Dict[str, float] = {}
        for name in selected:
            metric = self[name]
            if metric.aggregate == "ratio_of_sums":
                numerator = sum(self.value(metric.numerator, day) for day in records)
                denominator = sum(self.value(metric.denominator, day) for day in records)
                result[name] = numerator / denominator if denominator > 0 else 0.0
            else:
                values = [self.value(name, day) for day in records]
                if metric.aggregate == "mean" and not values:
                    raise ValueError(f"Cannot average empty days for {name!r}")
                result[name] = statistics.mean(values) if metric.aggregate == "mean" else sum(values)
            if not math.isfinite(result[name]):
                raise ValueError(f"Non-finite aggregate for {name!r}")
        return result


METRICS = MetricRegistry(
    (
        Metric("overflows", "count", "min", "Bin overflows"),
        Metric("kg", "kg", "max", "Waste collected"),
        Metric("ncol", "count", "min", "Bin collections"),
        Metric("kg_lost", "kg", "min", "Waste lost"),
        Metric("km", "km", "min", "Travel distance"),
        Metric("kg/km", "kg/km", "max", "Collection efficiency", "ratio_of_sums", "kg", "km"),
        Metric("profit", "EUR", "max", "Net profit"),
        Metric("time", "s", "min", "Policy compute time"),
        Metric("day_wall_time", "s", "min", "Day wall time"),
        Metric("time_spent", "h", "min", "Physical shift time"),
        Metric("days_simulated", "day", "none", "Days simulated"),
        Metric("collection_days", "day", "none", "Collection days"),
    )
)

# No mixed-unit reward or ambiguous `days` key. Shift time is opt-in for CPTPTC.
DEFAULT_METRICS: Tuple[str, ...] = tuple(name for name in METRICS if name != "time_spent")


@dataclass(frozen=True)
class SampleResult:
    """One sample's totals, including failed samples' partial diagnostic records."""

    sample_id: str
    metrics: Mapping[str, float]
    status: str = "completed"
    failed_day: Optional[int] = None
    exception_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in ("completed", "failed"):
            raise ValueError("Sample status must be completed or failed")


@dataclass(frozen=True)
class MetricSummary:
    """Equal-weight sample statistics; an empty successful population has no estimate."""

    mean: Mapping[str, Optional[float]]
    std: Mapping[str, Optional[float]]
    n_success: int
    n_failed: int
    n_ratio_defined: Mapping[str, int]


def summarize_samples(
    samples: Iterable[SampleResult],
    names: Iterable[str] = DEFAULT_METRICS,
    registry: MetricRegistry = METRICS,
) -> MetricSummary:
    """Exclude failed samples, retain zero-distance compatibility, count defined ratios.

    Completed zero-distance samples contribute ratio 0 to mean/std, matching the
    existing simulator. ``n_ratio_defined`` distinguishes that convention from a
    physically defined ratio. Partial failed metrics stay in the caller's records.
    """
    records = tuple(samples)
    if len({sample.sample_id for sample in records}) != len(records):
        raise ValueError("Duplicate sample IDs would bias the population summary")
    successful = [sample for sample in records if sample.status == "completed"]
    means: Dict[str, Optional[float]] = {}
    stds: Dict[str, Optional[float]] = {}
    defined: Dict[str, int] = {}
    for name in names:
        metric = registry[name]
        values = [registry.value(name, sample.metrics) for sample in successful]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite completed sample metric: {name!r}")
        means[name] = statistics.mean(values) if values else None
        stds[name] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        if metric.aggregate == "ratio_of_sums":
            defined[name] = sum(sample.metrics[metric.denominator] > 0 for sample in successful)
    return MetricSummary(means, stds, len(successful), len(records) - len(successful), defined)
