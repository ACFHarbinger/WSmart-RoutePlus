"""Tests for HpoHealthMetricsCallback and DEHB health penalty (§A.4 Option D)."""

from unittest.mock import MagicMock, patch

import pytest

optuna = pytest.importorskip("optuna")  # HPO is not in the 1.0.0 release environment
import torch  # noqa: E402 (after the optuna skip)
from logic.src.pipeline.callbacks.pytorch.hpo_health import (  # noqa: E402
    HpoHealthMetricsCallback,
    apply_dehb_health_penalty,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



def _make_trainer(epoch: int = 0, metrics: dict | None = None) -> MagicMock:
    trainer = MagicMock()
    trainer.is_global_zero = True
    trainer.current_epoch = epoch
    trainer.callback_metrics = metrics or {}
    return trainer


def test_apply_dehb_health_penalty_grad_explosion():
    penalised = apply_dehb_health_penalty(-5.0, grad_norm=150.0, entropy=0.05)
    assert penalised == 995.0


def test_apply_dehb_health_penalty_entropy_collapse():
    penalised = apply_dehb_health_penalty(-5.0, grad_norm=10.0, entropy=0.001)
    assert penalised == 495.0


def test_apply_dehb_health_penalty_healthy_trial():
    penalised = apply_dehb_health_penalty(-5.0, grad_norm=10.0, entropy=0.05)
    assert penalised == -5.0


def test_hpo_health_reports_optuna_user_attrs():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    cb = HpoHealthMetricsCallback(trial=trial, prune_on_unhealthy=False)
    trainer = _make_trainer(
        epoch=1,
        metrics={
            "train/grad_norm": torch.tensor(12.5),
            "train/entropy": torch.tensor(0.08),
            "val/reward": torch.tensor(-3.2),
        },
    )
    trainer.log_dir = "/tmp/lightning_logs/trial_0"

    cb.on_validation_epoch_end(trainer, MagicMock())

    assert trial.user_attrs["log_dir"] == "/tmp/lightning_logs/trial_0"
    assert trial.user_attrs["last_grad_norm"] == pytest.approx(12.5)
    assert trial.user_attrs["last_entropy"] == pytest.approx(0.08)
    assert trial.user_attrs["grad_norm_epoch_1"] == pytest.approx(12.5)
    assert trial.user_attrs["entropy_epoch_1"] == pytest.approx(0.08)


def test_hpo_health_prunes_on_grad_explosion():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    cb = HpoHealthMetricsCallback(trial=trial, max_grad_norm=50.0)
    trainer = _make_trainer(
        epoch=0,
        metrics={
            "train/grad_norm": torch.tensor(80.0),
            "train/entropy": torch.tensor(0.05),
            "val/reward": torch.tensor(-2.0),
        },
    )

    try:
        cb.on_validation_epoch_end(trainer, MagicMock())
        pruned = False
    except optuna.TrialPruned:
        pruned = True

    assert pruned
    assert trial.user_attrs["health_pruned"] == "grad_norm_explosion"


def test_hpo_health_logs_to_active_run():
    cb = HpoHealthMetricsCallback(prune_on_unhealthy=False)
    trainer = _make_trainer(
        epoch=2,
        metrics={
            "train/grad_norm": torch.tensor(4.0),
            "train/entropy": torch.tensor(0.12),
            "val/reward": torch.tensor(-1.5),
        },
    )
    mock_run = MagicMock()

    with patch(
        "logic.src.pipeline.callbacks.pytorch.hpo_health.get_active_run",
        return_value=mock_run,
    ):
        cb.on_validation_epoch_end(trainer, MagicMock())

    logged = {c.args[0]: c.args[1] for c in mock_run.log_metric.call_args_list}
    assert logged["hpo/grad_norm"] == pytest.approx(4.0)
    assert logged["hpo/entropy"] == pytest.approx(0.12)
    assert logged["hpo/val_reward"] == pytest.approx(-1.5)
