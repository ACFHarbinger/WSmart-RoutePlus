"""REINFORCE can scale the advantage by its batch spread so the loss ignores the reward unit."""

import pytest
import torch
from logic.src.pipeline.rl.core.reinforce import REINFORCE

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _Baseline:
    def eval(self, td, reward, env=None):
        return reward.mean()


def _loss(reward: torch.Tensor, normalize: bool) -> torch.Tensor:
    module = REINFORCE.__new__(REINFORCE)
    torch.nn.Module.__init__(module)
    module.entropy_weight = 0.0
    module.normalize_advantage = normalize
    module.baseline = _Baseline()
    module._current_baseline_val = None
    module.log = lambda *args, **kwargs: None
    log_likelihood = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    return module.calculate_loss(None, {"reward": reward, "log_likelihood": log_likelihood}, 0)


def test_normalised_loss_does_not_depend_on_the_reward_unit() -> None:
    reward = torch.tensor([1.0, -2.0, 0.5, 3.0])

    assert _loss(reward, True).item() == pytest.approx(_loss(reward * 400.0, True).item(), rel=1e-5)
    assert _loss(reward * 400.0, False).item() == pytest.approx(400.0 * _loss(reward, False).item(), rel=1e-5)
