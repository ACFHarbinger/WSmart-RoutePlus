"""
GPU memory monitoring callback for PyTorch Lightning (§F.2).

Tracks peak allocated and reserved GPU memory per epoch and optionally logs
memory summaries to identify allocation leaks during long RL training runs.

Attributes:
    GPUMemoryMonitor: Callback for GPU memory profiling.

Example:
    >>> from logic.src.pipeline.callbacks.pytorch.gpu_memory_monitor import GPUMemoryMonitor
    >>> trainer = pl.Trainer(callbacks=[GPUMemoryMonitor()])
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import Callback


class GPUMemoryMonitor(Callback):
    """
    Monitor GPU peak memory allocation and reservation per epoch (§F.2).

    Resets CUDA peak memory statistics at epoch start and logs peak allocated
    and reserved memory in Megabytes (MB) at epoch end.
    """

    def __init__(
        self,
        log_peak_allocated: bool = True,
        log_peak_reserved: bool = True,
        log_summary: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize GPUMemoryMonitor callback.

        Args:
            log_peak_allocated: Whether to log peak allocated memory (MB).
            log_peak_reserved: Whether to log peak reserved memory (MB).
            log_summary: Whether to log detailed torch.cuda.memory_summary().
            verbose: Whether to log peak memory stats via loguru.
        """
        super().__init__()
        self.log_peak_allocated = log_peak_allocated
        self.log_peak_reserved = log_peak_reserved
        self.log_summary = log_summary
        self.verbose = verbose

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Reset peak memory tracking at the start of each training epoch."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log peak GPU memory statistics at the end of each training epoch."""
        if not torch.cuda.is_available() or not trainer.is_global_zero:
            return

        device = pl_module.device
        if device.type != "cuda":
            return

        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

        if self.log_peak_allocated:
            pl_module.log(
                "memory/peak_allocated_mb",
                peak_alloc_mb,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

        if self.log_peak_reserved:
            pl_module.log(
                "memory/peak_reserved_mb",
                peak_res_mb,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

        if self.verbose:
            logger.info(
                f"[Epoch {trainer.current_epoch}] GPU Peak Allocated: {peak_alloc_mb:.2f} MB | "
                f"Peak Reserved: {peak_res_mb:.2f} MB (Device {device})"
            )

        if self.log_summary:
            summary = torch.cuda.memory_summary(device=device, abbreviated=True)
            logger.debug(f"GPU Memory Summary (Epoch {trainer.current_epoch}):\n{summary}")
