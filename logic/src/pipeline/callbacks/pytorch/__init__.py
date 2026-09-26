"""
Lightning Callbacks for WSmart-Route.

PyTorch Lightning Callback implementations for training monitoring,
meta-learning updates, speed measurement, GPU memory profiling, and model summaries.

Attributes:
    GPUMemoryMonitor: Tracks peak allocated and reserved GPU memory per epoch (§F.2).
    ModelSummaryCallback: Prints a rich table of model architecture details at training start.
    ReptileCallback: Outer-loop Reptile meta-learning update across tasks.
    SpeedMonitor: Logs forward/backward and data-loading times per step.
    TrainingDisplayCallback: Live terminal dashboard combining chart, metrics, and progress bars.
    TrainingHealthCallback: Automated health guardrails and instability detection.

Example:
    >>> from logic.src.pipeline.callbacks.pytorch import GPUMemoryMonitor
    >>> cb = GPUMemoryMonitor()
"""

from logic.src.pipeline.callbacks.pytorch.gpu_memory_monitor import GPUMemoryMonitor
from logic.src.pipeline.callbacks.pytorch.hpo_health import HpoHealthMetricsCallback
from logic.src.pipeline.callbacks.pytorch.model_summary import ModelSummaryCallback
from logic.src.pipeline.callbacks.pytorch.reptile import ReptileCallback
from logic.src.pipeline.callbacks.pytorch.speed_monitor import SpeedMonitor
from logic.src.pipeline.callbacks.pytorch.training_display import TrainingDisplayCallback
from logic.src.pipeline.callbacks.pytorch.training_health import TrainingHealthCallback

__all__ = [
    "GPUMemoryMonitor",
    "HpoHealthMetricsCallback",
    "ModelSummaryCallback",
    "ReptileCallback",
    "SpeedMonitor",
    "TrainingDisplayCallback",
    "TrainingHealthCallback",
]

