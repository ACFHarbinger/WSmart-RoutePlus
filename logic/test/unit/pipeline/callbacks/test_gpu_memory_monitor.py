"""Tests for GPUMemoryMonitor Lightning callback (§F.2)."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from logic.src.pipeline.callbacks.pytorch.gpu_memory_monitor import GPUMemoryMonitor

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_gpu_memory_monitor_init() -> None:
    """Test initialization parameters."""
    cb = GPUMemoryMonitor(
        log_peak_allocated=True,
        log_peak_reserved=True,
        log_summary=False,
        verbose=True,
    )
    assert cb.log_peak_allocated is True
    assert cb.log_peak_reserved is True
    assert cb.log_summary is False
    assert cb.verbose is True


def test_gpu_memory_monitor_epoch_start_no_cuda() -> None:
    """Test epoch start resets stats safely when CUDA is not available."""
    cb = GPUMemoryMonitor()
    trainer = MagicMock()
    pl_module = MagicMock()

    with patch("torch.cuda.is_available", return_value=False):
        cb.on_train_epoch_start(trainer, pl_module)
        # Should execute without error


def test_gpu_memory_monitor_epoch_start_with_cuda() -> None:
    """Test epoch start resets peak memory stats when CUDA is available."""
    cb = GPUMemoryMonitor()
    trainer = MagicMock()
    pl_module = MagicMock()

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.reset_peak_memory_stats"
    ) as mock_reset:
        cb.on_train_epoch_start(trainer, pl_module)
        mock_reset.assert_called_once()


def test_gpu_memory_monitor_epoch_end_logging() -> None:
    """Test peak allocated and reserved logging at epoch end."""
    cb = GPUMemoryMonitor(log_peak_allocated=True, log_peak_reserved=True, verbose=True)
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.current_epoch = 3

    pl_module = MagicMock()
    pl_module.device = torch.device("cuda:0")

    # 100 MB in bytes = 100 * 1024 * 1024
    bytes_100mb = 100 * 1024 * 1024
    bytes_200mb = 200 * 1024 * 1024

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.max_memory_allocated", return_value=bytes_100mb),
        patch("torch.cuda.max_memory_reserved", return_value=bytes_200mb),
    ):
        cb.on_train_epoch_end(trainer, pl_module)

        calls = pl_module.log.call_args_list
        assert len(calls) == 2

        # Check peak allocated log call
        args1, kwargs1 = calls[0]
        assert args1[0] == "memory/peak_allocated_mb"
        assert pytest.approx(args1[1], 0.01) == 100.0
        assert kwargs1["on_epoch"] is True

        # Check peak reserved log call
        args2, kwargs2 = calls[1]
        assert args2[0] == "memory/peak_reserved_mb"
        assert pytest.approx(args2[1], 0.01) == 200.0
        assert kwargs2["on_epoch"] is True


def test_gpu_memory_monitor_non_cuda_device_skipped() -> None:
    """Test that CPU training devices are skipped gracefully."""
    cb = GPUMemoryMonitor()
    trainer = MagicMock()
    trainer.is_global_zero = True
    pl_module = MagicMock()
    pl_module.device = torch.device("cpu")

    with patch("torch.cuda.is_available", return_value=True):
        cb.on_train_epoch_end(trainer, pl_module)
        pl_module.log.assert_not_called()
