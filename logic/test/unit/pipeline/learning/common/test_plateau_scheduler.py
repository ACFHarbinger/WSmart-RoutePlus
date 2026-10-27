"""The 'plateau' LR scheduler is returned in the form Lightning needs (a monitored metric)."""

import pytest
import torch
from logic.src.pipeline.learning.common.base.optimization import OptimizationMixin
from pytorch_lightning.core.optimizer import _configure_optimizers, _configure_schedulers_automatic_opt

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _Module(OptimizationMixin):
    def __init__(self, name, kwargs):
        self.policy = torch.nn.Linear(2, 1)
        self.optimizer_name = "adam"
        self.optimizer_kwargs = {"lr": 1e-3}
        self.lr_scheduler_name = name
        self.lr_scheduler_kwargs = kwargs


def _lightning_configs(module):
    _, schedulers, monitor = _configure_optimizers(module.configure_optimizers())
    return _configure_schedulers_automatic_opt(schedulers, monitor)


def test_plateau_monitors_validation_reward_and_maximizes_it():
    module = _Module("plateau", {"patience": 2})
    (config,) = _lightning_configs(module)
    assert isinstance(config.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert config.monitor == "val/reward"
    assert config.scheduler.mode == "max" and config.scheduler.patience == 2
    assert module.lr_scheduler_kwargs == {"patience": 2}  # the module's kwargs are not consumed


def test_plateau_respects_an_explicit_monitor_and_mode():
    (config,) = _lightning_configs(_Module("plateau", {"monitor": "val/cost", "mode": "min"}))
    scheduler = config.scheduler
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert config.monitor == "val/cost" and scheduler.mode == "min"


def test_other_schedulers_are_unchanged():
    (config,) = _lightning_configs(_Module("step", {"step_size": 3}))
    assert isinstance(config.scheduler, torch.optim.lr_scheduler.StepLR) and config.monitor is None
