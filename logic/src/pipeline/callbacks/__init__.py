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

from .pytorch.attention_heatmaps import AttentionHeatmapCallback
from .pytorch.gpu_memory_monitor import GPUMemoryMonitor
from .pytorch.hpo_health import HpoHealthMetricsCallback
from .pytorch.model_summary import ModelSummaryCallback
from .pytorch.reptile import ReptileCallback
from .pytorch.speed_monitor import SpeedMonitor
from .pytorch.training_display import TrainingDisplayCallback
from .pytorch.training_health import TrainingHealthCallback
from .simulation.policy_summary import PolicySummaryCallback
from .simulation.simulation_display import SimulationDisplayCallback

__all__ = [
    "AttentionHeatmapCallback",
    "GPUMemoryMonitor",
    "TrainingDisplayCallback",
    "HpoHealthMetricsCallback",
    "TrainingHealthCallback",
    "ReptileCallback",
    "SpeedMonitor",
    "ModelSummaryCallback",
    "SimulationDisplayCallback",
    "PolicySummaryCallback",
]
