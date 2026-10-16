"""WSTracker — centralized MLFlow-style experiment tracking for WSmart-Route.

This package provides the primary interface for tracking Experiments, Runs,
Metrics, and Parameters. It includes automated integrations for PyTorch
Lightning and manual logging capabilities for simulation and evaluation tasks.

Attributes:
    init: Primary entry-point to initialize the global Tracker.
    init_worker: Utility to attach a sub-process to an existing parent run.
    get_tracker: Accessor for the active global tracker instance.
    get_active_run: Accessor for the active process-local run.

Example:
    >>> import logic.src.tracking as wst
    >>> tracker = wst.init("My-Experiment")
    >>> with tracker.start_run("Run-1") as run:
    ...     run.log_metric("accuracy", 0.95)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from logic.src.constants import ROOT_DIR

from .core import tracker as _tracker_mod
from .core.run import Run, get_active_run
from .core.tracker import Tracker, get_tracker
from .integrations.data import RuntimeDataTracker
from .integrations.filesystem import FilesystemTracker
from .integrations.mlflow_bridge import MLflowBridge
from .integrations.simulation import SimulationRunTracker, get_sim_tracker
from .integrations.zenml_bridge import ZenMLBridge
from .viz_mixin import PolicyStateRecorder, PolicyVizMixin

__all__ = [
    # Factory / accessors
    "init",
    "init_worker",
    "get_tracker",
    "get_active_run",
    # Core classes
    "Tracker",
    "Run",
    # Integrations
    "TrackingCallback",
    "SimulationRunTracker",
    "get_sim_tracker",
    "RuntimeDataTracker",
    "FilesystemTracker",
    "MLflowBridge",
    "ZenMLBridge",
    # Submodules
    "hooks",
    "logging",
    "profiling",
    # Policy telemetry
    "PolicyVizMixin",
    "PolicyStateRecorder",
]

# ---------------------------------------------------------------------------
# Module-level initialisation helpers
# ---------------------------------------------------------------------------

_DEFAULT_TRACKING_URI = "assets/tracking"


def init(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    run_type: str = "generic",
    tags: Optional[Dict[str, str]] = None,
    description: str = "",
    buffer_size: int = 200,
) -> Tracker:
    """Initializes the global experiment Tracker.

    This is the primary entry-point for a main process (e.g., training). Calling
    this function configures the singleton tracker instance for the process.

    Args:
        experiment_name: Human-readable name of the experiment.
        tracking_uri: Optional directory path for the tracking database and
            artifacts. Defaults to 'assets/tracking' within the project root.
        run_type: Category identifier for runs created via this tracker.
        tags: Optional global tags to associate with the experiment.
        description: Textual description of the experiment.
        buffer_size: Default metric buffer size for runs created by this tracker.

    Returns:
        Tracker: The initialized process-global Tracker instance.
    """
    if tracking_uri is None:
        base_uri = "test_tracking" if os.environ.get("TEST_MODE") == "true" else _DEFAULT_TRACKING_URI
        tracking_uri = str(Path(ROOT_DIR).joinpath(base_uri).resolve())

    tracker = Tracker(tracking_uri=tracking_uri, buffer_size=buffer_size)
    _tracker_mod._tracker = tracker
    return tracker


def init_worker(
    tracking_uri: str,
    run_id: str,
    buffer_size: int = 200,
) -> Optional[Run]:
    """Attaches a worker sub-process to a specific parent Run.

    Typically used in parallel simulation workers to ensure metrics are
    logged back to the same run record without creating duplicate entries.

    Args:
        tracking_uri: The same database URI used by the parent process.
        run_id: The UUID of the active run created by the parent.
        buffer_size: Metric buffer capacity for this worker process.

    Returns:
        Optional[Run]: The attached Run instance if the ID exists, else None.
    """
    try:
        tracker = Tracker(tracking_uri=tracking_uri, buffer_size=buffer_size)
        _tracker_mod._tracker = tracker
        run = tracker.attach_run(run_id)
        return run
    except Exception:  # noqa: BLE001
        return None


def __getattr__(name):
    from logic.src.utils.registry import lazy_getattr

    return lazy_getattr(
        globals(),
        {
            "hooks": "logic.src.tracking.hooks",
            "logging": "logic.src.tracking.logging",
            "profiling": "logic.src.tracking.profiling",
            "TrackingCallback": "logic.src.tracking.integrations.lightning:TrackingCallback",
        },
        name,
    )
