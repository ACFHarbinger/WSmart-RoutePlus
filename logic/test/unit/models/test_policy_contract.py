"""The released Attention Model and the imitation expert keep the policy forward contract (P-24)."""

import pytest
import torch
from logic.src.envs.cmptp import CMPTPEnv
from logic.src.models.common.policy_contract import check_policy_output
from logic.src.models.core.attention_model import AttentionModel
from logic.src.policies.vectorized.routing.expert_cache import CachedExpert
from logic.src.policies.vectorized.routing.profitable_nearest_neighbour import ProfitableNearestNeighbourPolicy
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]

BATCH, BINS = 4, 8


def _setup() -> tuple[AttentionModel, CMPTPEnv, TensorDict]:
    torch.manual_seed(0)
    model = AttentionModel(embed_dim=32, hidden_dim=64, problem="cmptp", n_encode_layers=1, n_heads=4)
    model.eval()  # no dropout, so teacher forcing scores the sampled actions exactly
    env = CMPTPEnv(generator_params={"num_loc": BINS}, check_env_specs=False, waste_weight=10.0, cost_weight=1.0)
    g = torch.Generator().manual_seed(1)
    td = TensorDict(
        {
            "locs": torch.rand(BATCH, BINS, 2, generator=g),
            "depot": torch.full((BATCH, 2), 0.5),
            "waste": torch.rand(BATCH, BINS, generator=g),
            "capacity": torch.full((BATCH,), 2.0),
        },
        batch_size=[BATCH],
    )
    return model, env, td


def test_sampling_output_keeps_the_contract_and_carries_the_gradient() -> None:
    model, env, td = _setup()

    out = model(env.reset(td.clone()), env, strategy="sampling")

    check_policy_output(out, BATCH)
    assert out["log_likelihood"].requires_grad


def test_greedy_decoding_is_deterministic() -> None:
    model, env, td = _setup()

    first = model(env.reset(td.clone()), env, strategy="greedy")
    second = model(env.reset(td.clone()), env, strategy="greedy")

    check_policy_output(first, BATCH)
    assert torch.equal(first["actions"], second["actions"])


def test_teacher_forcing_scores_the_given_actions() -> None:
    model, env, td = _setup()
    sampled = model(env.reset(td.clone()), env, strategy="sampling")
    # Padding past the day end is not read.
    padded = torch.cat([sampled["actions"], torch.zeros(BATCH, 3, dtype=torch.long)], dim=1)

    forced = model(env.reset(td.clone()), env, actions=padded)

    check_policy_output(forced, BATCH)
    assert forced["log_likelihood"].requires_grad
    assert torch.allclose(forced["log_likelihood"], sampled["log_likelihood"], atol=1e-4)
    assert torch.allclose(forced["reward"], sampled["reward"])


@pytest.mark.parametrize("cached", [False, True])
def test_expert_output_is_teacher_forceable(tmp_path, cached: bool) -> None:
    model, env, td = _setup()
    expert = ProfitableNearestNeighbourPolicy("cmptp")
    if cached:
        expert = CachedExpert(expert, str(tmp_path))
    state = env.reset(td.clone())

    out = expert(state, env)
    check_policy_output(out, BATCH, needs_log_likelihood=False)

    forced = model(env.reset(td.clone()), env, actions=out["actions"])
    assert torch.isfinite(forced["log_likelihood"]).all()
    assert torch.allclose(forced["reward"], out["reward"])


def test_a_broken_output_is_reported() -> None:
    with pytest.raises(ValueError, match="lacks"):
        check_policy_output({"reward": torch.zeros(2)}, 2)
    with pytest.raises(ValueError, match="actions"):
        check_policy_output({"reward": torch.zeros(2), "actions": torch.zeros(2, 3)}, 2, needs_log_likelihood=False)
