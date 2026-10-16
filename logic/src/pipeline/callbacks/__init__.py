"""
Callbacks for WSmart-Route.

Provides PyTorch Lightning callbacks for training monitoring and
non-Lightning callbacks for simulation display.

Attributes:
    TrainingDisplayCallback: Rich/Plotext terminal dashboard for live training metrics.
    ReptileCallback: Reptile meta-learning outer-loop update callback.
    SpeedMonitor: Per-step and per-epoch timing monitor callback.
    ModelSummaryCallback: Detailed model architecture summary at training start.
    SimulationDisplayCallback: Real-time terminal dashboard for simulation runs.
    PolicySummaryCallback: Policy configuration summary before simulation.

Example:
    >>> from logic.src.pipeline.callbacks import TrainingDisplayCallback, SpeedMonitor
    >>> callbacks = [TrainingDisplayCallback(metric_keys="train/reward"), SpeedMonitor()]
"""

from logic.src.utils.registry import lazy_getattr

_LAZY = {
    "AttentionHeatmapCallback": "logic.src.pipeline.callbacks.pytorch.attention_heatmaps:AttentionHeatmapCallback",
    "GPUMemoryMonitor": "logic.src.pipeline.callbacks.pytorch.gpu_memory_monitor:GPUMemoryMonitor",
    "HpoHealthMetricsCallback": "logic.src.pipeline.callbacks.pytorch.hpo_health:HpoHealthMetricsCallback",
    "ModelSummaryCallback": "logic.src.pipeline.callbacks.pytorch.model_summary:ModelSummaryCallback",
    "ReptileCallback": "logic.src.pipeline.callbacks.pytorch.reptile:ReptileCallback",
    "SpeedMonitor": "logic.src.pipeline.callbacks.pytorch.speed_monitor:SpeedMonitor",
    "TrainingDisplayCallback": "logic.src.pipeline.callbacks.pytorch.training_display:TrainingDisplayCallback",
    "TrainingHealthCallback": "logic.src.pipeline.callbacks.pytorch.training_health:TrainingHealthCallback",
    "PolicySummaryCallback": "logic.src.pipeline.callbacks.simulation.policy_summary:PolicySummaryCallback",
    "SimulationDisplayCallback": "logic.src.pipeline.callbacks.simulation.simulation_display:SimulationDisplayCallback",
}

__all__ = list(_LAZY)


def __getattr__(name):
    return lazy_getattr(globals(), _LAZY, name)
