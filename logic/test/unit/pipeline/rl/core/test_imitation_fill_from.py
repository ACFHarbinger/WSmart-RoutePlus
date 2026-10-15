"""Whose routes empty the bins between time-training days during imitation (Q-23)."""

from typing import Any, List, Optional

import pytest
import torch
from logic.src.configs.rl.policies import ProfitableNNConfig
from logic.src.envs.tasks.profitable_tour.cptp import CPTPEnv
from logic.src.pipeline.rl.core.imitation import ImitationLearning
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _StubPolicy(torch.nn.Module):
    """Records its calls; a rollout visits bin 4 only, teacher forcing returns a likelihood."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls: List[str] = []

    def forward(self, td: TensorDict, env: Any, strategy: Optional[str] = None, actions=None, **kw: Any) -> dict:
        batch = td.batch_size[0]
        if actions is not None:
            self.calls.append("teacher")
            return {"log_likelihood": self.weight.expand(batch) - 1.0}
        self.calls.append(strategy or "")
        return {
            "reward": torch.zeros(batch),
            "actions": torch.tensor([[4, 0, 0]]).expand(batch, -1),
            "log_likelihood": torch.zeros(batch),
        }


def _module(fill_from: str, **kw: Any) -> tuple[ImitationLearning, TensorDict]:
    env = CPTPEnv(generator_params={"num_loc": 4}, check_env_specs=False, waste_weight=10.0, cost_weight=1.0)
    module = ImitationLearning(
        policy_config=ProfitableNNConfig(),
        env_name="cptp",
        fill_from=fill_from,
        env=env,
        policy=_StubPolicy(),
        train_time=True,
        **kw,
    )
    module.log = lambda *a, **k: None  # no trainer
    # Bins 1-3 sit together one unit from the depot and are worth a tour; bin 4 is far away.
    batch = TensorDict(
        {
            "locs": torch.tensor([[[1.0, 0.0], [1.0, 0.1], [1.1, 0.0], [0.0, 5.0]]]).expand(2, 4, 2).clone(),
            "depot": torch.zeros(2, 2),
            "waste": torch.tensor([[0.5, 0.5, 0.5, 0.1], [0.5, 0.5, 0.5, 0.1]]),
            "capacity": torch.full((2,), 10.0),
        },
        batch_size=[2],
    )
    return module, batch


def test_model_mode_keeps_the_sampled_rollout_and_its_actions_drive_the_fill() -> None:
    module, batch = _module("model")

    out = module.shared_step(batch, 0, phase="train")

    assert module.policy.calls == ["sampling", "teacher"]
    assert out["loss"].requires_grad
    assert set(module._epoch_actions[0].flatten().tolist()) == {0, 4}


def test_expert_mode_skips_the_rollout_and_the_expert_actions_drive_the_fill() -> None:
    module, batch = _module("expert")

    out = module.shared_step(batch, 0, phase="train")

    assert module.policy.calls == ["teacher"]
    assert out["source"] == "expert"
    assert out["loss"].requires_grad  # the teacher-forced pass keeps its gradient
    assert set(module._epoch_actions[0].flatten().tolist()) == {0, 1, 2, 3}


def test_validation_decodes_the_model_in_either_mode() -> None:
    module, batch = _module("expert")

    module.shared_step(batch, 0, phase="val")

    assert module.policy.calls == ["greedy"]


def test_unknown_fill_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="fill_from"):
        _module("sampled")


def test_precompute_fills_the_cache_and_trains_nothing(tmp_path) -> None:
    module, batch = _module("expert", expert_cache=str(tmp_path), precompute=True)

    out = module.shared_step(batch, 0, phase="train")
    module.expert_policy.flush()

    assert module.policy.calls == []
    assert out["loss"] is None
    assert set(module._epoch_actions[0].flatten().tolist()) == {0, 1, 2, 3}
    assert len(list(tmp_path.glob("shard-*.pt"))) == 1


def test_precompute_needs_the_expert_fill_and_a_cache(tmp_path) -> None:
    with pytest.raises(ValueError, match="precompute"):
        _module("model", expert_cache=str(tmp_path), precompute=True)
    with pytest.raises(ValueError, match="precompute"):
        _module("expert", precompute=True)
