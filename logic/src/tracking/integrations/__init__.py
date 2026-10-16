"""Pipeline integration helpers for the WSTracker ecosystem.

This package provides high-level trackers for various stages of the machine
learning pipeline, including in-memory data mutation tracking, filesystem
lineage, PyTorch Lightning callbacks, and simulation-specific instrumentation.

Attributes:
    TrackingCallback: PyTorch Lightning callback for run instrumentation.
    SimulationRunTracker: Specialized tracker for multi-day simulation episodes.
    get_sim_tracker: Retrieves the process-global simulation tracker instance.
    RuntimeDataTracker: Tracks in-memory dataset distribution mutations.
    FilesystemTracker: Tracks filesystem-level data events and lineage.
    ZenMLBridge: Integration bridge for ZenML pipeline tracking.

Example:
    >>> import logic.src.tracking.integrations as integrations
    >>> data_tracker = integrations.RuntimeDataTracker(run)
    >>> fs_tracker = integrations.FilesystemTracker(run)
"""

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "RuntimeDataTracker": "logic.src.tracking.integrations.data:RuntimeDataTracker",
    "FilesystemTracker": "logic.src.tracking.integrations.filesystem:FilesystemTracker",
    "TrackingCallback": "logic.src.tracking.integrations.lightning:TrackingCallback",
    "SimulationRunTracker": "logic.src.tracking.integrations.simulation:SimulationRunTracker",
    "get_sim_tracker": "logic.src.tracking.integrations.simulation:get_sim_tracker",
    "ZenMLBridge": "logic.src.tracking.integrations.zenml_bridge:ZenMLBridge",
}

__all__ = list(_LAZY)


def __getattr__(name):
    return lazy_getattr(globals(), _LAZY, name)
