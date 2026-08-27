"""
Data drift detection using Evidently AI.

Compares a synthetic evaluation dataset against an empirical real-world baseline
distribution and writes a standalone HTML report to disk.

Supported file formats for both reference and current datasets:

* ``.csv``   — loaded directly with ``pandas.read_csv``.
* ``.npz``   — each array key is summarised per-instance (mean, std, min, max)
  to produce a 2-D tabular representation suitable for column-wise drift tests.
* ``.json`` / ``.jsonl`` — loaded via ``pandas.read_json``.
* ``.pkl``   — must contain a ``pandas.DataFrame`` or plain ``dict``.

CLI usage::

    python -m logic.src.pipeline.features.eval.drift_detection \\
        --reference assets/datasets/april_2024_summary.csv \\
        --current   assets/datasets/eval_synthetic.npz \\
        --output_dir assets/drift_reports/ \\
        --problem vrpp \\
        --stattest ks

Programmatic usage::

    from logic.src.pipeline.features.eval.drift_detection import run_drift_detection

    report_path = run_drift_detection(
        reference_path="assets/datasets/april_2024_summary.csv",
        current_path="assets/datasets/eval_synthetic.npz",
        problem="vrpp",
    )
    print(f"Report saved to: {report_path}")

Attributes:
    run_drift_detection: Runs drift detection between a reference and current dataset.
    _load_npz_as_df: Load an NPZ file into a 2-D DataFrame for drift analysis.
    _load_json_or_jsonl_as_df: Load a JSON or JSONL file into a DataFrame.
    _load_and_flatten: Load a dataset from disk (CSV, NPZ, JSON/JSONL, or PKL) and flatten it into a 2-D DataFrame.
    _check_evidently: Check if Evidently AI is installed.

Example:
    >>> from logic.src.pipeline.features.eval.drift_detection import run_drift_detection
    >>>
    >>> report_path = run_drift_detection(
    ...     reference_path="assets/datasets/april_2024_summary.csv",
    ...     current_path="assets/datasets/eval_synthetic.npz",
    ...     problem="vrpp",
    ... )
    >>> print(f"Report saved to: {report_path}")
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import evidently  # type: ignore[import]
    from evidently import ColumnMapping  # type: ignore[import]
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset  # type: ignore[import]
    from evidently.metrics import (  # type: ignore[import]
        ColumnDriftMetric,
        ColumnSummaryMetric,
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
    )
    from evidently.report import Report  # type: ignore[import]
except ImportError:
    evidently = None  # type: ignore[assignment]
    ColumnMapping = None  # type: ignore[assignment]
    DataDriftPreset = None  # type: ignore[assignment]
    DataQualityPreset = None  # type: ignore[assignment]
    ColumnDriftMetric = None  # type: ignore[assignment]
    ColumnSummaryMetric = None  # type: ignore[assignment]
    DatasetDriftMetric = None  # type: ignore[assignment]
    DatasetMissingValuesMetric = None  # type: ignore[assignment]
    Report = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_drift_detection(
    reference_path: str,
    current_path: str,
    output_dir: str = "assets/drift_reports",
    report_filename: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    problem: str = "vrpp",
    stattest: str = "ks",
    stattest_threshold: float = 0.05,
) -> str:
    """
    Run Evidently drift detection between a reference and current dataset.

    Both datasets are normalised to flat 2-D DataFrames, aligned on shared
    columns, and passed to ``evidently.report.Report`` for analysis.  The
    resulting report is saved as a standalone HTML file.

    Args:
        reference_path: Path to the empirical / real-world baseline dataset
            (CSV, NPZ, JSON, JSONL, or PKL).
        current_path: Path to the synthetic evaluation dataset.
        output_dir: Directory where the HTML report will be written.
        report_filename: Override the output filename.  Defaults to
            ``"drift_report_{problem}_{timestamp}.html"``.
        feature_columns: Explicit list of feature column names to include.
            When None, all shared numeric columns (excluding ``target_column``)
            are used automatically.
        target_column: Optional target/label column name for target drift
            analysis (e.g. ``"total_cost"``).
        problem: Problem type tag embedded in the default report filename.
            One of ``"vrpp"``, ``"wcvrp"``, ``"sdwcvrp"``, ``"all"``.
        stattest: Statistical test for numeric columns.
            Options: ``"ks"`` (Kolmogorov-Smirnov, default), ``"psi"``
            (Population Stability Index), ``"wasserstein"``, ``"mannw"``.
        stattest_threshold: p-value / threshold below which drift is flagged.
            Default 0.05.

    Returns:
        Absolute path to the generated HTML report file.

    Raises:
        ImportError: If ``evidently`` is not installed.
        FileNotFoundError: If either dataset file does not exist.
        ValueError: If no shared columns are found between the two datasets.
    """
    _check_evidently()

    # ── Load datasets ────────────────────────────────────────────────────────
    reference_df = load_and_flatten(reference_path, problem=problem)
    current_df = load_and_flatten(current_path, problem=problem)

    log.info(
        "Reference: %d rows × %d cols | Current: %d rows × %d cols",
        len(reference_df),
        len(reference_df.columns),
        len(current_df),
        len(current_df.columns),
    )

    # ── Align on shared columns ──────────────────────────────────────────────
    shared = sorted(set(reference_df.columns) & set(current_df.columns))
    if not shared:
        raise ValueError(
            "No shared columns between reference and current datasets. "
            f"Reference cols (first 10): {list(reference_df.columns)[:10]}. "
            f"Current cols (first 10): {list(current_df.columns)[:10]}."
        )

    numeric_shared = [
        c
        for c in shared
        if reference_df[c].dtype in (np.float64, np.float32, np.int64, np.int32) and c != target_column
    ]

    if feature_columns:
        numeric_shared = [c for c in feature_columns if c in numeric_shared]

    log.info("Feature columns for drift analysis (%d): %s", len(numeric_shared), numeric_shared)

    reference_df = reference_df[shared].copy()
    current_df = current_df[shared].copy()

    # ── Column mapping ───────────────────────────────────────────────────────
    col_map = ColumnMapping()
    col_map.numerical_features = numeric_shared
    if target_column and target_column in shared:
        col_map.target = target_column

    # ── Build report ─────────────────────────────────────────────────────────
    metrics = [
        DataDriftPreset(stattest=stattest, stattest_threshold=stattest_threshold),
        DataQualityPreset(),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
    ]
    if target_column and target_column in shared:
        metrics.append(ColumnDriftMetric(column_name=target_column))

    report = Report(metrics=metrics)
    log.info("Running Evidently analysis (stattest=%s, threshold=%s) ...", stattest, stattest_threshold)
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=col_map,
    )

    # ── Save report ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = report_filename or f"drift_report_{problem}_{timestamp}.html"
    out_path = str(Path(output_dir) / fname)
    report.save_html(out_path)
    log.info("Drift report saved → %s", out_path)

    _log_drift_summary(report)
    return os.path.abspath(out_path)


def run_column_drift_suite(
    reference_path: str,
    current_path: str,
    columns: Optional[List[str]] = None,
    problem: str = "vrpp",
    output_dir: str = "assets/drift_reports",
    stattest: str = "ks",
    max_columns: int = 10,
) -> str:
    """
    Run a focused per-column drift analysis for a subset of features.

    Produces a smaller, faster HTML report that includes one
    ``ColumnDriftMetric`` and one ``ColumnSummaryMetric`` per target column.

    Args:
        reference_path: Reference dataset path (CSV or NPZ).
        current_path: Current dataset path (CSV or NPZ).
        columns: Specific column names to test.  When None, the first
            ``max_columns`` shared numeric columns are used.
        problem: Problem type for the report filename. Default ``"vrpp"``.
        output_dir: Output directory for the HTML report.
        stattest: Statistical test identifier. Default ``"ks"``.
        max_columns: Maximum number of columns when ``columns`` is None.
            Default 10.

    Returns:
        Absolute path to the generated HTML report.
    """
    _check_evidently()

    reference_df = load_and_flatten(reference_path, problem=problem)
    current_df = load_and_flatten(current_path, problem=problem)

    shared = sorted(set(reference_df.columns) & set(current_df.columns))
    if columns:
        target_cols = [c for c in columns if c in shared]
    else:
        target_cols = [c for c in shared if reference_df[c].dtype != object][:max_columns]

    metrics = []
    for col in target_cols:
        metrics.append(ColumnDriftMetric(column_name=col, stattest=stattest))
        metrics.append(ColumnSummaryMetric(column_name=col))

    report = Report(metrics=metrics)
    report.run(
        reference_data=reference_df[shared],
        current_data=current_df[shared],
        column_mapping=ColumnMapping(),
    )

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(output_dir) / f"column_drift_{problem}_{timestamp}.html")
    report.save_html(out_path)
    log.info("Column drift report saved → %s", out_path)
    return os.path.abspath(out_path)


# ---------------------------------------------------------------------------
# Dataset loading helpers (public for reuse in tests)
# ---------------------------------------------------------------------------


def load_and_flatten(path: str, problem: str = "vrpp") -> pd.DataFrame:
    """
    Load a dataset from disk and return a flat 2-D pandas DataFrame.

    CSV / JSON / JSONL files are loaded as-is.
    NPZ files are summarised: each array key produces ``{key}_mean``,
    ``{key}_std``, ``{key}_min``, ``{key}_max`` columns.

    Args:
        path: Absolute or relative path to the dataset file.
        problem: Problem type hint (used only for logging). Default ``"vrpp"``.

    Returns:
        2-D DataFrame ready for Evidently drift analysis.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported.
    """
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = fp.suffix.lower()
    log.debug("Loading %s dataset from %s", problem, path)

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".npz", ".npy"):
        df = _npz_to_dataframe(path)
    elif ext == ".json":
        df = pd.read_json(path)
    elif ext == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, pd.DataFrame):
            df = obj
        elif isinstance(obj, dict):
            df = pd.DataFrame(obj)
        else:
            raise ValueError(f"Unsupported pickle content type: {type(obj)}. Expected DataFrame or dict.")
    else:
        raise ValueError(f"Unsupported file extension: '{ext}'. Supported: .csv, .npz, .npy, .json, .jsonl, .pkl")

    df = df.dropna(axis=1, how="all")
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _npz_to_dataframe(path: str) -> pd.DataFrame:
    """
    Load an NPZ file and compute per-instance summary statistics.

    Arrays of shape ``(N,)`` → one column.
    Arrays of shape ``(N, D)`` → four columns: mean, std, min, max.
    Arrays of shape ``(N, K, D)`` → flattened to ``(N, K*D)`` then summarised.
    Scalar arrays are skipped.

    Args:
        path: Path to the NPZ file.

    Returns:
        A 2-D DataFrame containing the summary statistics.
    """
    data = np.load(path, allow_pickle=True)
    rows: Dict[str, np.ndarray] = {}
    n_instances: Optional[int] = None

    for key in data.files:
        arr = data[key]

        if arr.ndim == 0:
            continue

        if arr.ndim == 1:
            rows[key] = arr
            n_instances = n_instances or len(arr)

        elif arr.ndim == 2:
            rows[f"{key}_mean"] = arr.mean(axis=1)
            rows[f"{key}_std"] = arr.std(axis=1)
            rows[f"{key}_min"] = arr.min(axis=1)
            rows[f"{key}_max"] = arr.max(axis=1)
            n_instances = n_instances or arr.shape[0]

        elif arr.ndim == 3:
            flat = arr.reshape(arr.shape[0], -1)
            rows[f"{key}_mean"] = flat.mean(axis=1)
            rows[f"{key}_std"] = flat.std(axis=1)
            n_instances = n_instances or arr.shape[0]

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def _check_evidently() -> None:
    """Raise a helpful ImportError if evidently is not installed."""
    if evidently is None:
        raise ImportError("evidently is required for drift detection. Run: pip install evidently")


def _log_drift_summary(report: Any) -> None:
    """Print a brief drift summary to the logger (non-critical).

    Args:
        report: The generated drift report.
    """
    try:
        result = report.as_dict()
        metrics = result.get("metrics", [])
        drifted = sum(1 for m in metrics if m.get("result", {}).get("drift_detected") is True)
        total = sum(1 for m in metrics if "drift_detected" in m.get("result", {}))
        if total > 0:
            log.info(
                "Drift detected in %d / %d features (%.1f%%)",
                drifted,
                total,
                100.0 * drifted / total,
            )
    except Exception:
        pass  # Summary is non-critical


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser for drift detection.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run Evidently data drift detection between a reference (real-world) and current (synthetic) dataset."
        )
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the reference (empirical / real-world) dataset.",
    )
    parser.add_argument(
        "--current",
        required=True,
        help="Path to the current (synthetic / evaluation) dataset.",
    )
    parser.add_argument(
        "--output_dir",
        default="assets/drift_reports",
        help="Directory for HTML report output. Default: assets/drift_reports",
    )
    parser.add_argument(
        "--problem",
        default="vrpp",
        choices=["vrpp", "cvrpp", "ttop", "wcvrp", "sdwcvrp", "all"],
        help="Problem type tag used in the report filename. Default: vrpp",
    )
    parser.add_argument(
        "--target_column",
        default=None,
        help="Optional target column name for target drift analysis.",
    )
    parser.add_argument(
        "--stattest",
        default="ks",
        choices=["ks", "psi", "wasserstein", "mannw"],
        help="Statistical test for numeric columns. Default: ks",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Statistical test threshold for drift flagging. Default: 0.05",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main() -> None:
    """CLI entry point for data drift detection."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    report_path = run_drift_detection(
        reference_path=args.reference,
        current_path=args.current,
        output_dir=args.output_dir,
        problem=args.problem,
        target_column=args.target_column,
        stattest=args.stattest,
        stattest_threshold=args.threshold,
    )
    print(f"\nDrift report saved: {report_path}")


if __name__ == "__main__":
    main()
