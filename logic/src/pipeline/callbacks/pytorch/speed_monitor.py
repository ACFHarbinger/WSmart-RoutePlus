"""Speed monitoring callback for Lightning.

Attributes:
    SpeedMonitor: Callback to monitor training speed.

Example:
    >>> from logic.src.pipeline.callbacks.pytorch.speed_monitor import SpeedMonitor
    >>> trainer = L.Trainer(callbacks=[SpeedMonitor()])
"""

import time
from typing import Optional

import pytorch_lightning as L
from lightning_fabric.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.parsing import AttributeDict


class SpeedMonitor(Callback):
    """
    Monitor the speed of each step and each epoch.
    Logs intra-step (forward/backward) and inter-step (data loading) times.

    Attributes:
        _snap_intra_step_time: Timestamp of intra-step measurement start.
        _snap_inter_step_time: Timestamp of inter-step measurement start.
        _snap_epoch_time: Timestamp of epoch measurement start.
        _log_stats: AttributeDict containing logging flags.
        verbose: Whether to print timing info to console.
    """

    def __init__(
        self,
        intra_step_time: bool = True,
        inter_step_time: bool = True,
        epoch_time: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize SpeedMonitor callback.

        Args:
            intra_step_time: Whether to log forward/backward time.
            inter_step_time: Whether to log data loading time.
            epoch_time: Whether to log epoch duration.
            verbose: Whether to print timing info to console.
        """
        super().__init__()
        self._log_stats = AttributeDict(
            {
                "intra_step_time": intra_step_time,
                "inter_step_time": inter_step_time,
                "epoch_time": epoch_time,
            }
        )
        self.verbose = verbose
        self._snap_intra_step_time: Optional[float] = None
        self._snap_inter_step_time: Optional[float] = None
        self._snap_epoch_time: Optional[float] = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset epoch time snapshot at training start.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module instance.
        """
        self._snap_epoch_time = None

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset timing snapshots and capture epoch start time.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module instance.
        """
        self._snap_intra_step_time = None
        self._snap_inter_step_time = None
        self._snap_epoch_time = time.time()

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset inter-step time snapshot for validation.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module instance.
        """
        self._snap_inter_step_time = None

    def on_test_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Reset inter-step time snapshot for testing.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module instance.
        """
        self._snap_inter_step_time = None

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        *unused_args,
        **unused_kwargs,
    ) -> None:
        """Capture intra-step start time and log inter-step time.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            unused_args: Unused arguments.
            unused_kwargs: Unused keyword arguments.
        """
        if self._log_stats.intra_step_time:
            self._snap_intra_step_time = time.time()

        if not self._should_log(trainer):
            return

        logs: dict[str, float] = {}
        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs["time/inter_step_ms"] = (time.time() - self._snap_inter_step_time) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        *unused_args,
        **unused_kwargs,
    ) -> None:
        """Capture inter-step start time and log intra-step time.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module instance.
            unused_args: Unused arguments.
            unused_kwargs: Unused keyword arguments.
        """
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if self.verbose and self._log_stats.intra_step_time and self._snap_intra_step_time:
            pl_module.print(f"time/intra_step_ms: {(time.time() - self._snap_intra_step_time) * 1000}")

        if not self._should_log(trainer):
            return

        logs: dict[str, float] = {}
        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            logs["time/intra_step_ms"] = (time.time() - self._snap_intra_step_time) * 1000

        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Log epoch duration at end of training epoch.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module instance.
        """
        logs: dict[str, float] = {}
        if self._log_stats.epoch_time and self._snap_epoch_time:
            logs["time/epoch_s"] = time.time() - self._snap_epoch_time
        if trainer.logger is not None:
            trainer.logger.log_metrics(logs, step=trainer.global_step)

    @staticmethod
    def _should_log(trainer: L.Trainer) -> bool:
        """Determine whether to log based on current step and logging frequency.

        Args:
            trainer: The PyTorch Lightning trainer instance.

        Returns:
            bool: True if logging should occur, False otherwise.
        """
        return (trainer.global_step + 1) % getattr(trainer, "log_every_n_steps", 1) == 0 or trainer.should_stop
