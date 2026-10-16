"""Paired comparison primitives for the Q-14 regression gates (WP-12, #117).

All functions here are pure (JSON-like inputs only): they never touch the
simulator, the filesystem, or the clock, so they are unit-testable without
any heavy run. Heavy execution lives in ``run_gate.py``.

Conventions (owner rulings Q-14, RQ-codex-2, RQ-codex-3, RQ-kimi-1):

* Gate (a) compares routes, states and KPIs bit for bit, excluding timing
  and run metadata (``EXCLUDED_KEYS``).
* Gate (b) compares ``kgkm`` and ``overflows`` against frozen tolerances.
  A result outside the tolerances in *either* direction needs review.
* Across samples, ``kgkm`` stays the per-sample ratio of sums with an
  equal-weight mean/std (RQ-codex-3); pooling across samples is out of scope.
* The BPC cell may use certificate equivalence (incumbent objective and
  certified bound within 1e-6) instead of bit identity (RQ-kimi-1).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

# Exclude measured compute/wall time, but retain physical shift hours (time_spent).
EXCLUDED_KEYS = frozenset(
    {
        "time",
        "day_wall_time",
        "execution_time",
        "timestamp",
        "solver_certificate",  # Proof metadata is compared explicitly in certificate fallback.
        "run_id",
        "hostname",
        "pid",
        "seed",  # recorded config echo, not a result
    }
)

# Owner ruling RQ-kimi-1: certificate equivalence tolerance.
CERT_TOL = 1e-6


class Diff(NamedTuple):
    """One non-excluded difference between two result trees."""

    path: str
    expected: Any
    actual: Any


class Verdict(NamedTuple):
    """Outcome of a tolerance or certificate comparison."""

    passed: bool
    detail: str


def strip_excluded(obj: Any) -> Any:
    """Return a copy of ``obj`` with every ``EXCLUDED_KEYS`` entry removed."""
    if isinstance(obj, dict):
        return {k: strip_excluded(v) for k, v in obj.items() if k not in EXCLUDED_KEYS}
    if isinstance(obj, (list, tuple)):
        return [strip_excluded(v) for v in obj]
    return obj


def _values_equal(expected: Any, actual: Any) -> bool:
    """Bit-for-bit equality with NaN == NaN (missing sensor data)."""
    if isinstance(expected, float) and isinstance(actual, float) and math.isnan(expected) and math.isnan(actual):
        return True
    return type(expected) is type(actual) and expected == actual


def _iter_diffs(expected: Any, actual: Any, path: str) -> List[Diff]:
    diffs: List[Diff] = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected:
            if key not in actual:
                diffs.append(Diff(f"{path}.{key}", expected[key], "<missing>"))
            else:
                diffs.extend(_iter_diffs(expected[key], actual[key], f"{path}.{key}"))
        for key in actual:
            if key not in expected:
                diffs.append(Diff(f"{path}.{key}", "<missing>", actual[key]))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(Diff(f"{path}.<len>", len(expected), len(actual)))
        for i, (e, a) in enumerate(zip(expected, actual, strict=False)):
            diffs.extend(_iter_diffs(e, a, f"{path}[{i}]"))
    elif not _values_equal(expected, actual):
        diffs.append(Diff(path, expected, actual))
    return diffs


def compare_exact(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> List[Diff]:
    """Bit-for-bit compare of two per-policy log trees, timing excluded.

    Args:
        baseline: Reference ``log_<policy>_*N.json`` content (samples + daily).
        candidate: Same structure from the candidate run.

    Returns:
        Diffs with paths; empty means the gate passes.
    """
    return _iter_diffs(strip_excluded(baseline), strip_excluded(candidate), "$")


def extract_cell_kpis(sample_metrics: Dict[str, Any]) -> Tuple[float, float]:
    """Return ``(kgkm, overflows)`` from a samples-section metric dict."""
    return float(sample_metrics["kg/km"] if "kg/km" in sample_metrics else sample_metrics["kgkm"]), float(
        sample_metrics["overflows"]
    )


def compare_tolerance(
    baseline_kgkm: float,
    baseline_overflows: float,
    candidate_kgkm: float,
    candidate_overflows: float,
    tol_kgkm: float,
    tol_overflows: float,
) -> Verdict:
    """Gate-(b) check: absolute deltas within the frozen tolerances.

    Either direction outside tolerance fails (better results need review too,
    per RQ-codex-2).
    """
    values = (baseline_kgkm, baseline_overflows, candidate_kgkm, candidate_overflows, tol_kgkm, tol_overflows)
    if not all(math.isfinite(v) for v in values) or tol_kgkm < 0 or tol_overflows < 0:
        return Verdict(False, "metrics and nonnegative tolerances must be finite")
    d_kgkm = abs(candidate_kgkm - baseline_kgkm)
    d_over = abs(candidate_overflows - baseline_overflows)
    passed = d_kgkm <= tol_kgkm and d_over <= tol_overflows
    return Verdict(
        passed,
        f"|d_kgkm|={d_kgkm:.6g} (tol {tol_kgkm:.6g}), |d_overflows|={d_over:.6g} (tol {tol_overflows:.6g})",
    )


def compare_certificates(
    baseline_incumbent: float,
    baseline_bound: float,
    candidate_incumbent: float,
    candidate_bound: float,
    tol: float = CERT_TOL,
) -> Verdict:
    """RQ-kimi-1 certificate equivalence for the exact-solver cell.

    Passes when incumbent objectives agree within ``tol`` and both bounds are
    valid certificates (bound >= incumbent on a maximisation objective).
    """
    if (
        not all(
            math.isfinite(v) for v in (baseline_incumbent, baseline_bound, candidate_incumbent, candidate_bound, tol)
        )
        or tol < 0
    ):
        return Verdict(False, "incumbents, certified bounds and nonnegative tolerance must be finite")
    d_inc = abs(candidate_incumbent - baseline_incumbent)
    d_bound = abs(candidate_bound - baseline_bound)
    valid_base = baseline_bound + tol >= baseline_incumbent
    valid_cand = candidate_bound + tol >= candidate_incumbent
    passed = d_inc <= tol and d_bound <= tol and valid_base and valid_cand
    return Verdict(
        passed,
        f"|d_incumbent|={d_inc:.3g}, |d_bound|={d_bound:.3g} (tol {tol:.3g}), base_bound_valid={valid_base}, cand_bound_valid={valid_cand}",
    )


def freeze_tolerances(repeats: List[Dict[str, float]]) -> Dict[str, float]:
    """Derive frozen tolerances from repeated unchanged-code runs (RQ-codex-2).

    Args:
        repeats: One ``{"kgkm": .., "overflows": ..}`` dict per repeat run of
            the same cell on the unchanged baseline code.

    Returns:
        ``{"tol_kgkm": max pairwise |Δ|, "tol_overflows": max pairwise |Δ|}``.
        Calibrated on the unchanged code only, frozen before any refactor
        result is seen.
    """
    if len(repeats) < 2:
        raise ValueError("At least two unchanged-code repeats are required")
    if not all(math.isfinite(float(row[key])) for row in repeats for key in ("kgkm", "overflows")):
        raise ValueError("Calibration metrics must be finite")
    tol_kgkm = 0.0
    tol_over = 0.0
    for i in range(len(repeats)):
        for j in range(i + 1, len(repeats)):
            tol_kgkm = max(tol_kgkm, abs(repeats[i]["kgkm"] - repeats[j]["kgkm"]))
            tol_over = max(tol_over, abs(repeats[i]["overflows"] - repeats[j]["overflows"]))
    return {"tol_kgkm": tol_kgkm, "tol_overflows": tol_over}


def check_repeats_identical(repeats: List[Dict[str, Any]]) -> List[Diff]:
    """Prove the gate-(a) config is deterministic: repeats must match exactly.

    Returns the diffs of every repeat against the first; non-empty means the
    cell's "timing off" config is not actually deterministic and the cell
    must use certificate/tolerance mode instead of bit identity.
    """
    diffs: List[Diff] = []
    for n, rep in enumerate(repeats[1:], start=2):
        for d in compare_exact(repeats[0], rep):
            diffs.append(Diff(f"repeat{n}:{d.path}", d.expected, d.actual))
    return diffs


def load_tolerances(path: str) -> Dict[str, Any]:
    """Load a frozen tolerances file written by ``run_gate.py calibrate``."""
    import json

    with open(path) as f:
        return json.load(f)


def gate_b_verdict(
    archived_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    tolerances: Dict[str, float],
    candidate_label: Optional[str] = None,
) -> Verdict:
    """One-cell gate-(b) verdict from summary rows plus frozen tolerances."""
    label = f"[{candidate_label}] " if candidate_label else ""
    try:
        verdict = compare_tolerance(
            *extract_cell_kpis(archived_row),
            *extract_cell_kpis(candidate_row),
            float(tolerances["tol_kgkm"]),
            float(tolerances["tol_overflows"]),
        )
    except (KeyError, ValueError, TypeError) as e:
        return Verdict(False, f"{label}missing key in summary row: {e}")
    return Verdict(verdict.passed, f"{label}{verdict.detail}")


def compare_daily_certificates(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Verdict:
    """Compare every explicitly paired sample/day proof; never aggregate bounds."""
    left = baseline.get("daily")
    right = candidate.get("daily")
    samples = baseline.get("samples", {})
    if not isinstance(left, dict) or not isinstance(right, dict) or not samples:
        return Verdict(False, "Missing per-sample daily certificate evidence")
    if set(left) != set(right) or set(left) != set(samples) or set(samples) != set(candidate.get("samples", {})):
        return Verdict(False, "Sample keys differ between results and daily proofs")
    for sample in sorted(left):
        base, cand = left[sample], right[sample]
        days = base.get("day")
        if not isinstance(days, list) or not days or days != cand.get("day") or len(set(days)) != len(days):
            return Verdict(False, f"Missing, duplicate or unpaired days for sample {sample}")
        proofs = base.get("solver_certificate")
        other = cand.get("solver_certificate")
        if (
            not isinstance(proofs, list)
            or not isinstance(other, list)
            or len(proofs) != len(days)
            or len(other) != len(days)
        ):
            return Verdict(False, f"Incomplete daily proofs for sample {sample}")
        for day, a, b in zip(days, proofs, other):  # noqa: B905 - lengths validated above; Python 3.9
            if not isinstance(a, dict) or not isinstance(b, dict):
                return Verdict(False, f"Uncertified sample {sample}, day {day}")
            if not a.get("instance") or a.get("instance") != b.get("instance"):
                return Verdict(False, f"Construction inputs differ for sample {sample}, day {day}")
            try:
                verdict = compare_certificates(
                    float(a["incumbent"]), float(a["bound"]), float(b["incumbent"]), float(b["bound"]), CERT_TOL
                )
            except (KeyError, TypeError, ValueError):
                return Verdict(False, f"Malformed proof for sample {sample}, day {day}")
            if not verdict.passed:
                return Verdict(False, f"Sample {sample}, day {day}: {verdict.detail}")
    return Verdict(True, "Every sample/day incumbent and certified bound agrees")


_FLOAT_KPIS = {"overflows", "kg", "ncol", "kg_lost", "km", "kg/km", "profit",
               "time_spent", "days_simulated", "collection_days"}


def _canonical_quantities(row: Dict[str, Any]) -> None:
    for key in _FLOAT_KPIS & row.keys():
        value = row[key]
        row[key] = [float(item) for item in value] if isinstance(value, list) else float(value)


def _canonical_legacy_metrics(row: Dict[str, Any]) -> None:
    row.pop("reward", None)
    if "days" in row:
        old = row.pop("days")
        if "collection_days" in row and old != row["collection_days"]:
            raise ValueError("Conflicting legacy days and collection_days")
        row["collection_days"] = old
    _canonical_quantities(row)


def canonical_gate_log(log: Dict[str, Any]) -> Dict[str, Any]:
    """Translate only approved P-21/P-42 wire-schema changes on an archived log.

    Retired reward is removed; days is explicitly collection_days. Horizon days
    come from the paired day column. Physical scalar KPIs use the new float
    representation. Routes, statuses, states and numeric values are not relaxed.
    """
    result = copy.deepcopy(log)
    daily = result.get("daily")
    if not isinstance(daily, dict):
        return result
    horizons = {}
    for sample, row in daily.items():
        days = row.get("day", [])
        if not isinstance(days, list) or not days:
            continue
        horizons[sample] = len(days)
        row.pop("reward", None)
        row.setdefault("days_simulated", [1.0] * len(days))
        if "collection_days" not in row and isinstance(row.get("ncol"), list):
            row["collection_days"] = [float(value > 0) for value in row["ncol"]]
        _canonical_quantities(row)
    for sample, row in result.get("samples", {}).items():
        if sample in horizons:
            row.setdefault("days_simulated", float(horizons[sample]))
        _canonical_legacy_metrics(row)
    for section in ("mean", "std"):
        row = result.get(section)
        if not isinstance(row, dict):
            continue
        if horizons:
            values = list(horizons.values())
            mean = sum(values) / len(values)
            row.setdefault("days_simulated", mean if section == "mean" else
                           math.sqrt(sum((value - mean)**2 for value in values) / len(values)))
        _canonical_legacy_metrics(row)
    return result
