"""Resume adaptive imitation without restarting its annealing state."""

from types import SimpleNamespace

import pytest
import torch
from logic.src.pipeline.learning.core.adaptive_imitation import AdaptiveImitation

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _module(monkeypatch):
    monkeypatch.setattr(AdaptiveImitation, "_create_expert_policy", lambda *args: torch.nn.Identity())
    module = AdaptiveImitation(
        policy_config=None,
        env_name="cptp",
        env=SimpleNamespace(),
        policy=torch.nn.Linear(1, 1),
        baseline="none",
        patience=5,
    )
    module.log = lambda *args, **kwargs: None
    module._trainer = SimpleNamespace(callback_metrics={"val/reward": torch.tensor(9.0)}, current_epoch=10)
    return module


@pytest.mark.parametrize("stopped", [False, True])
def test_resume_preserves_schedule_and_next_epoch_decision(monkeypatch, tmp_path, stopped):
    original = _module(monkeypatch)
    original.current_il_weight = 0.95**10
    original.stop_il = stopped
    original.wait = 4
    original.best_reward = 10.0
    checkpoint = {"state_dict": original.state_dict()}
    original.on_save_checkpoint(checkpoint)
    path = tmp_path / "resume.ckpt"
    torch.save(checkpoint, path)

    resumed = _module(monkeypatch)
    saved = torch.load(path, weights_only=False)
    resumed.load_state_dict(saved["state_dict"])
    resumed.on_load_checkpoint(saved)
    assert resumed.current_il_weight == original.current_il_weight
    assert resumed.stop_il is stopped
    assert resumed.wait == 4
    assert resumed.best_reward == 10.0

    original.on_train_epoch_end()
    resumed.on_train_epoch_end()
    assert resumed.current_il_weight == original.current_il_weight
    assert resumed.stop_il == original.stop_il
    assert resumed.wait == original.wait
    assert resumed.best_reward == original.best_reward
    if not stopped:
        assert resumed.current_il_weight == resumed.initial_il_weight  # fifth rejection reheats


def test_legacy_checkpoint_keeps_constructor_defaults(monkeypatch):
    module = _module(monkeypatch)
    module.on_load_checkpoint({})
    assert module.current_il_weight == module.initial_il_weight
    assert module.stop_il is False
    assert module.wait == 0
    assert module.best_reward == float("-inf")
