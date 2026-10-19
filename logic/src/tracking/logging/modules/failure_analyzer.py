"""Simulation failure root-cause analysis (§A.6).

Compares predicted vs. actual bin fill levels after each simulation day,
flags overflow contributors and skipped high-fill bins, and produces a
structured summary for JSON logs and the WSmart-Route Studio.

Example:
    >>> analyzer = FailureAnalyzer()
    >>> summary = analyzer.analyze(
    ...     new_overflows=1,
    ...     sum_lost=5.0,
    ...     profit=-2.0,
    ...     fill=np.array([12.0]),
    ...     total_fill=np.array([100.0]),
    ...     bins_means=np.array([4.0]),
    ...     bins_real_c=np.array([100.0]),
    ...     tour=[0],
    ...     collected=np.zeros(1),
    ...     coords=pd.DataFrame({"ID": [101]}),
    ... )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from logic.src.constants import MAX_CAPACITY_PERCENT

FILL_SPIKE_RATIO = 2.0
HIGH_FILL_THRESHOLD = 80.0


class FailureAnalyzer:
    """Post-day simulation failure diagnostics."""

    def analyze(
        self,
        *,
        new_overflows: int,
        sum_lost: float,
        profit: float,
        fill: np.ndarray,
        total_fill: np.ndarray,
        bins_means: np.ndarray,
        bins_real_c: np.ndarray,
        tour: Sequence[int],
        collected: Optional[np.ndarray],
        coords: Union[pd.DataFrame, List[Any]],
        mandatory: Optional[Sequence[int]] = None,
        fill_spike_ratio: float = FILL_SPIKE_RATIO,
        high_fill_threshold: float = HIGH_FILL_THRESHOLD,
    ) -> Dict[str, Any]:
        """Build a structured failure summary for one simulation day.

        Args:
            new_overflows: Number of bins at capacity after today's fill.
            sum_lost: Waste lost to overflow today (kg).
            profit: Net profit for the day (€).
            fill: Ground-truth daily fill increments per bin.
            total_fill: Observed fill levels after deposition.
            bins_means: Running mean daily fill rate per bin.
            bins_real_c: Real fill levels after collection.
            tour: Route node indices (0 = depot).
            collected: Per-bin collected mass vector.
            coords: Bin coordinate/metadata table.
            mandatory: Mandatory node indices before routing.
            fill_spike_ratio: Actual/predicted ratio above which a spike is flagged.
            high_fill_threshold: Fill % considered high-risk when skipped.

        Returns:
            Structured summary dict with ``has_failure``, root causes, and bin lists.
        """
        root_causes = self._detect_root_causes(new_overflows, sum_lost, profit)
        has_failure = bool(root_causes)

        visited = {node - 1 for node in tour if node > 0}
        mandatory_set = {m - 1 for m in (mandatory or []) if m > 0}
        collected_arr = np.asarray(collected if collected is not None else np.zeros_like(fill))

        overflow_bins = self._overflow_bins(
            fill=fill,
            bins_means=bins_means,
            bins_real_c=bins_real_c,
            visited=visited,
            collected_arr=collected_arr,
            coords=coords,
            fill_spike_ratio=fill_spike_ratio,
            root_causes=root_causes,
        )
        skipped_high_fill = self._skipped_high_fill_bins(
            total_fill=total_fill,
            visited=visited,
            mandatory_set=mandatory_set,
            coords=coords,
            high_fill_threshold=high_fill_threshold,
        )

        if skipped_high_fill and "skipped_high_fill" not in root_causes:
            root_causes.append("skipped_high_fill")

        severity = self._severity(new_overflows, sum_lost, profit, len(overflow_bins))

        return {
            "has_failure": has_failure,
            "severity": severity,
            "root_causes": root_causes,
            "summary": self._summary_message(root_causes, new_overflows, sum_lost, profit),
            "metrics": {
                "new_overflows": new_overflows,
                "kg_lost": sum_lost,
                "profit": profit,
            },
            "overflow_bins": overflow_bins,
            "skipped_high_fill_bins": skipped_high_fill,
        }

    @staticmethod
    def _detect_root_causes(new_overflows: int, sum_lost: float, profit: float) -> List[str]:
        causes: List[str] = []
        if new_overflows > 0:
            causes.append("overflow_event")
        if sum_lost > 0:
            causes.append("waste_lost")
        if profit < 0:
            causes.append("negative_profit")
        return causes

    @staticmethod
    def _severity(new_overflows: int, sum_lost: float, profit: float, n_overflow_bins: int) -> str:
        if new_overflows >= 3 or sum_lost >= 50.0 or (profit < 0 and new_overflows > 0):
            return "critical"
        if new_overflows > 0 or sum_lost > 0 or profit < 0:
            return "warning"
        if n_overflow_bins > 0:
            return "info"
        return "ok"

    @staticmethod
    def _summary_message(causes: List[str], new_overflows: int, sum_lost: float, profit: float) -> str:
        if not causes:
            return "No simulation failures detected for this day."
        parts: List[str] = []
        if "overflow_event" in causes:
            parts.append(f"{new_overflows} bin(s) reached capacity")
        if "waste_lost" in causes:
            parts.append(f"{sum_lost:.1f} kg waste lost to overflow")
        if "negative_profit" in causes:
            parts.append(f"negative profit ({profit:.2f} €)")
        if "skipped_high_fill" in causes:
            parts.append("high-fill bins were skipped by the route")
        if "fill_rate_spike" in causes:
            parts.append("fill rate exceeded statistical prediction")
        return "; ".join(parts).capitalize() + "."

    def _overflow_bins(
        self,
        *,
        fill: np.ndarray,
        bins_means: np.ndarray,
        bins_real_c: np.ndarray,
        visited: set[int],
        collected_arr: np.ndarray,
        coords: Union[pd.DataFrame, List[Any]],
        fill_spike_ratio: float,
        root_causes: List[str],
    ) -> List[Dict[str, Any]]:
        if not isinstance(bins_real_c, np.ndarray):
            try:
                bins_real_c = np.asarray(bins_real_c)
                if bins_real_c.dtype == object:
                    bins_real_c = np.zeros(0)
            except Exception:
                bins_real_c = np.zeros(0)
        overflow_mask = bins_real_c >= MAX_CAPACITY_PERCENT
        indices = np.nonzero(overflow_mask)[0]
        results: List[Dict[str, Any]] = []

        for idx in indices:
            predicted = float(bins_means[idx]) if idx < len(bins_means) else 0.0
            actual = float(fill[idx]) if idx < len(fill) else 0.0
            spike = predicted > 1e-6 and actual / predicted >= fill_spike_ratio
            if spike and "fill_rate_spike" not in root_causes:
                root_causes.append("fill_rate_spike")

            was_collected = float(collected_arr[idx]) > 0 if idx < len(collected_arr) else False
            results.append(
                {
                    "bin_index": int(idx),
                    "bin_id": self._resolve_bin_id(idx, coords),
                    "predicted_fill": predicted,
                    "actual_fill": actual,
                    "fill_delta": actual - predicted,
                    "fill_spike": spike,
                    "in_tour": int(idx) in visited,
                    "collected": was_collected,
                    "fill_level_after": float(bins_real_c[idx]),
                }
            )
        return results

    def _skipped_high_fill_bins(
        self,
        *,
        total_fill: np.ndarray,
        visited: set[int],
        mandatory_set: set[int],
        coords: Union[pd.DataFrame, List[Any]],
        high_fill_threshold: float,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx, level in enumerate(total_fill):
            if float(level) < high_fill_threshold:
                continue
            if idx in visited:
                continue
            results.append(
                {
                    "bin_index": idx,
                    "bin_id": self._resolve_bin_id(idx, coords),
                    "fill_level": float(level),
                    "mandatory": idx in mandatory_set,
                }
            )
        results.sort(key=lambda row: row["fill_level"], reverse=True)
        return results[:10]

    @staticmethod
    def _resolve_bin_id(bin_index: int, coords: Union[pd.DataFrame, List[Any]]) -> int:
        node_idx = bin_index + 1
        if isinstance(coords, pd.DataFrame) and node_idx < len(coords):
            try:
                return int(coords.iloc[node_idx]["ID"])
            except (KeyError, TypeError, ValueError):
                pass
        return bin_index
