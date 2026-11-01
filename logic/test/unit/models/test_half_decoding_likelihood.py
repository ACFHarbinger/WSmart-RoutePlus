"""Explicit half-precision decoding keeps exact likelihood and masked gradients."""

import pytest
import torch
from logic.src.utils.decoding.greedy import Greedy
from logic.src.utils.decoding.sampling import Sampling


@pytest.mark.parametrize("strategy", [Sampling, Greedy])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_logits_match_float32_log_probability_and_gradient(strategy, dtype):
    logits = torch.zeros(1, 100, dtype=dtype, requires_grad=True)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[:, -1] = False
    torch.manual_seed(3)
    action, score, entropy = strategy().step(logits, mask)
    assert action.item() != 99
    reference = logits.float().masked_fill(~mask, -torch.inf).log_softmax(-1)
    expected = reference.gather(1, action[:, None]).squeeze(-1)
    torch.testing.assert_close(score, expected)
    grad = torch.autograd.grad(score.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0]
    torch.testing.assert_close(grad, expected_grad)
    if strategy is Sampling:
        assert entropy.item() == pytest.approx(torch.tensor(99.0).log().item())
        entropy_grad = torch.autograd.grad(entropy.sum(), logits)[0]
        assert torch.isfinite(entropy_grad).all()
        assert entropy_grad[0, -1].item() == 0


def test_half_teacher_forcing_rare_action_matches_log_softmax(monkeypatch):
    from logic.src.envs.cmptp import CMPTPEnv
    from logic.src.models.core.attention_model.policy import AttentionModelPolicy

    env = CMPTPEnv(generator_params={"num_loc": 2})
    policy = AttentionModelPolicy(env_name="cmptp", embed_dim=16, n_heads=2, n_encode_layers=1)
    logits = torch.tensor([[0.0, -11.5, 0.0]], dtype=torch.float16, requires_grad=True)
    monkeypatch.setattr(policy.decoder, "_get_log_p", lambda *args: (logits, torch.zeros(1, 3, dtype=torch.bool)))
    output = policy(env.reset(batch_size=[1]), env=env, actions=torch.tensor([[1]]))
    expected = logits.float().log_softmax(-1)[:, 1]
    torch.testing.assert_close(output["log_likelihood"], expected)
    actual_grad = torch.autograd.grad(output["log_likelihood"].sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
