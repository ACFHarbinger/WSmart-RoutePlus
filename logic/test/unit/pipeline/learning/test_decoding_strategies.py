

import pytest
import torch
from logic.src.utils.decoding import BeamSearch, Evaluate, Greedy, Sampling, top_k_filter, top_p_filter
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]





def test_greedy_decoding():
    strategy = Greedy()
    logits = torch.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True], [True, True, True]], dtype=torch.bool)

    action, log_prob, entropy = strategy.step(logits, mask)

    assert torch.all(action == torch.tensor([1, 0]))
    assert torch.allclose(entropy, torch.zeros_like(entropy))

    # Test with mask
    mask = torch.tensor([[True, False, True], [True, True, True]], dtype=torch.bool)
    action, log_prob, entropy = strategy.step(logits, mask)
    assert action[0] == 2  # 5.0 is masked


def test_sampling_decoding():
    torch.manual_seed(42)
    strategy = Sampling(temperature=1.0)
    logits = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    # With uniform logits, actions should be spread
    actions = []
    for _ in range(100):
        action, _, _ = strategy.step(logits, mask)
        actions.append(action.item())

    counts = torch.bincount(torch.tensor(actions))
    assert counts.shape[0] == 3
    assert counts.min() > 20  # Roughly uniform


def test_beam_search_decoding():
    strategy = BeamSearch(beam_width=2)
    logits = torch.tensor([[1.0, 5.0, 2.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    action, log_prob, entropy = strategy.step(logits, mask)

    assert action.shape == (1, 2)
    assert torch.all(action == torch.tensor([[1, 2]]))  # Top 2 are indices 1 and 2


def test_evaluate_decoding():
    actions = torch.tensor([[1, 2]])
    strategy = Evaluate(actions=actions)
    logits = torch.tensor([[1.0, 5.0, 2.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    # First step
    action, log_prob, entropy = strategy.step(logits, mask)
    assert action.item() == 1

    # Second step
    logits2 = torch.tensor([[4.0, 3.0, 6.0]], dtype=torch.float32)
    action, log_prob, entropy = strategy.step(logits2, mask)
    assert action.item() == 2


def test_top_k_filter():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]], dtype=torch.float32)
    filtered = top_k_filter(logits, k=2)
    assert filtered[0, 1] == 5.0
    assert filtered[0, 3] == 4.0
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 2] == float("-inf")


def test_top_p_filter():
    # Probs: 0.1, 0.7, 0.2
    logits = torch.tensor(
        [[torch.log(torch.tensor(0.1)), torch.log(torch.tensor(0.7)), torch.log(torch.tensor(0.2))]],
        dtype=torch.float32,
    )

    # top_p = 0.8 should keep 0.7 and 0.2 (sum=0.9 > 0.8)
    filtered = top_p_filter(logits, p=0.8)
    assert filtered[0, 1] == logits[0, 1]
    assert filtered[0, 2] == logits[0, 2]
    assert filtered[0, 0] == float("-inf")


def test_multistart_hooks():
    strategy = Sampling(multistart=True, num_starts=3, select_best=True)
    td = TensorDict({"reward": torch.tensor([1.0, 1.0])}, batch_size=[2])

    # pre_decoder_hook
    td_expanded, _, num_starts = strategy.pre_decoder_hook(td, None)
    assert td_expanded.batch_size[0] == 6
    assert num_starts == 3

    # post_decoder_hook (simulated rewards)
    td_expanded["reward"] = torch.tensor(
        [
            1.0,
            5.0,
            2.0,  # Batch 1 -> best is index 1
            4.0,
            3.0,
            1.0,
        ]
    )  # Batch 2 -> best is index 0
    actions = torch.tensor([[10], [11], [12], [20], [21], [22]])
    log_probs = torch.tensor([-1.0, -0.5, -2.0, -1.0, -2.0, -3.0])

    final_ll, final_actions, final_td, _ = strategy.post_decoder_hook(td_expanded, None, log_probs, actions)

    assert final_actions.shape == (2, 1)
    assert final_actions[0].item() == 11
    assert final_actions[1].item() == 20
    assert final_td.batch_size[0] == 2
