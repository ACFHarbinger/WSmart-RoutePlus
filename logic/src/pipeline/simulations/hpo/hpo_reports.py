"""Optuna study visualization export for post-hoc HPO analysis (§A.5).

Renders ``optuna.visualization`` Plotly figures and writes HTML artefacts to
``assets/hpo_reports/`` for offline sharing and Studio inspection.

Example:
    >>> from logic.src.pipeline.simulations.hpo.hpo_reports import export_optuna_study_reports
    >>> report_dir = export_optuna_study_reports(study)
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import optuna

from logic.src.constants import ROOT_DIR

try:
    import optuna.visualization as ov
except ImportError:
    ov = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_REPORTS_SUBDIR = os.path.join("assets", "hpo_reports")
MIN_COMPLETED_FOR_PLOTS = 2


def _safe_study_slug(study_name: str) -> str:
    """Return a filesystem-safe slug for a study name."""
    slug = re.sub(r"[^\w.\-]+", "_", study_name.strip())
    return slug or "study"


def _write_plotly_figure(fig: Any, stem: str, report_dir: str) -> List[str]:
    """Write a Plotly figure to HTML (and PNG when kaleido is available)."""
    written: List[str] = []
    html_path = os.path.join(report_dir, f"{stem}.html")
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    written.append(html_path)

    try:
        import kaleido  # noqa: F401  # pyrefly: ignore[missing-import]  # optional plot export

        png_path = os.path.join(report_dir, f"{stem}.png")
        fig.write_image(png_path, scale=2)
        written.append(png_path)
    except Exception:
        pass

    return written


def export_optuna_study_reports(
    study: optuna.Study,
    output_dir: Optional[str] = None,
    *,
    min_completed: int = MIN_COMPLETED_FOR_PLOTS,
) -> Optional[str]:
    """Export parallel-coordinates, importances, and history plots for a study.

    Args:
        study: Loaded Optuna study.
        output_dir: Parent directory for report folders. Defaults to
            ``<ROOT_DIR>/assets/hpo_reports``.
        min_completed: Minimum completed trials required to emit plots.

    Returns:
        Path to the report directory, or ``None`` when export was skipped.
    """
    if ov is None:
        logger.warning("optuna.visualization unavailable — skipping HPO report export.")
        return None

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < min_completed:
        logger.info(
            "HPO report export skipped for %s (%d completed trials; need %d).",
            study.study_name,
            len(completed),
            min_completed,
        )
        return None

    parent = output_dir or os.path.join(str(ROOT_DIR), DEFAULT_REPORTS_SUBDIR)
    os.makedirs(parent, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(parent, f"{_safe_study_slug(study.study_name)}_{stamp}")
    os.makedirs(report_dir, exist_ok=True)

    artefacts: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}

    plot_specs = [
        ("parallel_coordinate", lambda: ov.plot_parallel_coordinate(study)),
        ("param_importances", lambda: ov.plot_param_importances(study)),
        ("optimization_history", lambda: ov.plot_optimization_history(study)),
    ]

    all_files: List[str] = []
    for stem, builder in plot_specs:
        try:
            fig = builder()
            files = _write_plotly_figure(fig, stem, report_dir)
            artefacts[stem] = [os.path.basename(f) for f in files]
            all_files.extend(files)
        except Exception as exc:
            errors[stem] = str(exc)
            logger.warning("HPO report plot %s failed: %s", stem, exc)

    if not artefacts:
        logger.warning("No HPO report artefacts written for study %s.", study.study_name)
        return None

    manifest = {
        "study_name": study.study_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "best_value": study.best_value if completed else None,
        "best_params": study.best_params if completed else {},
        "artefacts": artefacts,
        "errors": errors,
    }
    manifest_path = os.path.join(report_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    all_files.append(manifest_path)

    logger.info("HPO reports exported to %s (%d files).", report_dir, len(all_files))
    return report_dir


def export_optuna_study_from_storage(
    storage_url: str,
    study_name: str,
    output_dir: Optional[str] = None,
    *,
    min_completed: int = MIN_COMPLETED_FOR_PLOTS,
) -> Optional[str]:
    """Load a study from Optuna storage and export visualization reports.

    Args:
        storage_url: SQLAlchemy storage URL (e.g. ``sqlite:///assets/hpo/study.db``).
        study_name: Study name inside the storage backend.
        output_dir: Optional parent directory for report folders.
        min_completed: Minimum completed trials required to emit plots.

    Returns:
        Path to the report directory, or ``None`` when export was skipped.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    return export_optuna_study_reports(
        study,
        output_dir=output_dir,
        min_completed=min_completed,
    )
