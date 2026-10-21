"""Runtime and memory prediction for cloud cells (CR-11).

**Wall time.** Three gradient-boosted quantile regressions (10th, 50th and 90th
percentile, scikit-learn) on the logarithm of a cell's wall time, with one-hot
routing algorithm, packing variant, improver, distribution and area, and numeric
network size and horizon. The bundled table (``runtime_training.csv``, built by
``tools/build_runtime_training.py``) holds the paper's 810 runs on one laptop:
210 cells timed by the rerun drivers and 600 whose wall time is their policy time
plus the measured simulator overhead. Completed cloud tasks can be added
(:func:`.history.cloud_records`).

**Machines.** The table comes from one machine, so the model predicts laptop time;
a machine type's speed factor is the median ratio of observed to predicted time of
its completed tasks (at least ``MIN_MACHINE_SAMPLES``), 1.0 otherwise.

**Memory.** The 90th percentile of observed task peaks (CR-12 ``memory_rss_mb``) of the
same routing algorithm and network size, times a 1.25 margin, once
``MIN_MEMORY_SAMPLES`` exist; otherwise the rerun drivers' reservation
(1.5 GiB plus 2.0 GiB at 350 bins or more, 1.2 GiB below).

**Fallback.** Without scikit-learn or for a cell with no size or horizon, the
cell's ``expected_wall_seconds`` with a 0.5x-2x interval, or None.

Attributes:
    Prediction: One cell's prediction.
    RuntimePredictor: Fit, predict and evaluate.
    default_predictor: A predictor fitted on the bundled table (cached).
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from logic.controllers.cloud.model import CellSpec
from logic.controllers.cloud.prediction.features import CATEGORICAL, NUMERIC, cell_features

TRAINING_TABLE = Path(__file__).with_name("runtime_training.csv")
QUANTILES = (0.1, 0.5, 0.9)
MODEL_CATEGORICAL = [c for c in CATEGORICAL if c != "machine_type"]
MIN_MACHINE_SAMPLES = 3
MIN_MEMORY_SAMPLES = 3


@dataclass
class Prediction:
    """One cell's predicted wall time (seconds) and peak memory (GiB).

    Attributes:
        cell_id: The cell.
        p10, p50, p90: Wall-time quantiles in seconds (None when unknown).
        memory_gb: Memory to reserve.
        source: ``model``, ``expected`` (the manifest's estimate) or ``none``.
        memory_source: ``observed`` or ``heuristic``.
        machine_factor: Speed factor applied for the machine type.
    """

    cell_id: str
    p10: Optional[float]
    p50: Optional[float]
    p90: Optional[float]
    memory_gb: float
    source: str
    memory_source: str
    machine_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_table(path: Path = TRAINING_TABLE) -> List[Dict[str, Any]]:
    """Training rows: feature columns, ``wall_seconds`` and ``source``."""
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            row["size"], row["horizon"] = int(row["size"]), int(row["horizon"])
            row["wall_seconds"] = float(row["wall_seconds"])
            rows.append(row)
    return rows


def heuristic_memory_gb(size: Optional[int]) -> float:
    """The rerun drivers' per-cell reservation."""
    return 1.5 + (2.0 if (size or 0) >= 350 else 1.2)


class RuntimePredictor:
    """Quantile wall-time model, machine speed factors and memory estimates.

    Args:
        rows: Training rows (see :func:`load_table`); records with ``wall_seconds <= 0`` are ignored.
        machine_factors: Speed factor per machine type.
        memory_peaks: Observed peak memory in MiB per ``(policy, size)``.
        random_state: Seed of the boosting subsamples.
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        machine_factors: Optional[Dict[str, float]] = None,
        memory_peaks: Optional[Dict[tuple, List[float]]] = None,
        random_state: int = 0,
    ) -> None:
        self.rows = [r for r in rows if float(r["wall_seconds"]) > 0]
        self.machine_factors = dict(machine_factors or {})
        self.memory_peaks = dict(memory_peaks or {})
        self.random_state = random_state
        self.models: Optional[Dict[float, Any]] = None
        self.error: Optional[str] = None
        try:
            self.models = self._fit(self.rows)
        except ImportError as exc:  # scikit-learn missing: predictions fall back to the manifest
            self.error = f"{type(exc).__name__}: {exc}"

    def _frame(self, rows: Sequence[Dict[str, Any]]):
        import pandas as pd

        def _val(r, k):
            v = r.get(k)
            if k == "policy" and v in ("na.amgat", "amgat", "na"):
                return "am"
            return v

        return pd.DataFrame([{k: _val(r, k) for k in MODEL_CATEGORICAL + NUMERIC} for r in rows])

    def _fit(self, rows: Sequence[Dict[str, Any]]) -> Dict[float, Any]:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        X = self._frame(rows)
        y = [math.log(float(r["wall_seconds"])) for r in rows]
        models = {}
        for q in QUANTILES:
            encode = ColumnTransformer(
                [("cat", OneHotEncoder(handle_unknown="ignore"), MODEL_CATEGORICAL)], remainder="passthrough"
            )
            boost = GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
            )
            models[q] = Pipeline([("encode", encode), ("boost", boost)]).fit(X, y)
        return models

    def predict_features(self, features: Sequence[Dict[str, Any]]) -> List[Optional[List[float]]]:
        """Laptop-time ``[p10, p50, p90]`` per feature dict, or None where the model cannot predict."""
        usable = [i for i, f in enumerate(features) if f.get("size") is not None and f.get("horizon") is not None]
        out: List[Optional[List[float]]] = [None] * len(features)
        if self.models is None or not usable:
            return out
        X = self._frame([features[i] for i in usable])
        by_q = {q: m.predict(X) for q, m in self.models.items()}
        for row, i in enumerate(usable):
            out[i] = sorted(math.exp(float(by_q[q][row])) for q in QUANTILES)  # no quantile crossing
        return out

    def memory_gb(self, policy: str, size: Optional[int]) -> tuple:
        peaks = self.memory_peaks.get((policy, size), [])
        if len(peaks) >= MIN_MEMORY_SAMPLES:
            p90 = sorted(peaks)[min(len(peaks) - 1, math.ceil(0.9 * len(peaks)) - 1)]
            return round(max(0.5, 1.25 * p90 / 1024), 2), "observed"
        return heuristic_memory_gb(size), "heuristic"

    def predict(self, cells: Sequence[CellSpec], machine_type: Optional[str] = None) -> List[Prediction]:
        """Predictions for ``cells`` run on ``machine_type``."""
        features = [cell_features(c, machine_type) for c in cells]
        factor = self.machine_factors.get(machine_type or "", 1.0)
        out = []
        for cell, f, q in zip(cells, features, self.predict_features(features), strict=True):
            memory, memory_source = self.memory_gb(f["policy"], f["size"])
            if q is not None:
                p10, p50, p90 = (round(v * factor, 1) for v in q)
                out.append(Prediction(cell.cell_id, p10, p50, p90, memory, "model", memory_source, factor))
            elif cell.expected_wall_seconds:
                e = float(cell.expected_wall_seconds)
                out.append(Prediction(cell.cell_id, 0.5 * e, e, 2.0 * e, memory, "expected", memory_source, 1.0))
            else:
                out.append(Prediction(cell.cell_id, None, None, None, memory, "none", memory_source, 1.0))
        return out

    def evaluate(self, holdout_fraction: float = 0.25, seed: int = 0) -> Dict[str, Any]:
        """Refit without a random share of the directly timed rows and score on that share.

        Only ``rerun_wall`` rows are held out, since their wall time was measured; the
        ``policy_time`` rows are always kept for training.
        """
        import random

        timed = [r for r in self.rows if r.get("source") == "rerun_wall"]
        rng = random.Random(seed)
        held = set(rng.sample(range(len(timed)), max(1, int(len(timed) * holdout_fraction)))) if timed else set()
        test = [timed[i] for i in sorted(held)]
        held_ids = {id(r) for r in test}
        train = [r for r in self.rows if id(r) not in held_ids]
        model = RuntimePredictor(train, random_state=self.random_state)
        if model.models is None or not test:
            return {"holdout": len(test), "error": model.error or "no held-out rows"}
        quantiles = model.predict_features(test)
        ape, inside, abs_err = [], 0, []
        for row, q in zip(test, quantiles, strict=True):
            actual = float(row["wall_seconds"])
            ape.append(abs(q[1] - actual) / actual)
            abs_err.append(abs(q[1] - actual))
            inside += q[0] <= actual <= q[2]
        return {
            "train_rows": len(train),
            "holdout": len(test),
            "median_ape": round(statistics.median(ape), 4),
            "mean_ape": round(statistics.fmean(ape), 4),
            "median_abs_error_s": round(statistics.median(abs_err), 1),
            "p10_p90_coverage": round(inside / len(test), 3),
            "holdout_policies": sorted({r["policy"] for r in test}),
        }


@lru_cache(maxsize=1)
def default_predictor() -> RuntimePredictor:
    """A predictor fitted on the bundled training table."""
    return RuntimePredictor(load_table())
