"""NeuralAgent.compute_batch_sim: dict outputs, per-instance distance matrices, hook cleanup."""

from types import SimpleNamespace

import pytest
import torch
from logic.src.policies.routing.learning_algorithms.neural_agent.agent import NeuralAgent
from torch import nn

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class _Attn(nn.Module):
    def forward(self, x):
        self.last_attn = (torch.ones(1), torch.zeros(1))
        return x


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.att = SimpleNamespace(module=_Attn())
        self.inner = self.att.module

    def forward(self, x):
        return self.att.module(x)


class _Model(nn.Module):
    def __init__(self, pi):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_Layer()])
        self.pi = pi
        self.problem = SimpleNamespace(
            evaluate_tour=lambda inp, pi, cw_dict=None: {
                "cost": torch.zeros(pi.size(0)),
                "overflows": torch.zeros(pi.size(0)),
                "waste": torch.zeros(pi.size(0)),
            }
        )

    def forward(self, input, **kwargs):
        self.encoder.layers[0](torch.zeros(1))
        return {"actions": self.pi, "cost": torch.zeros(self.pi.size(0))}


def _hooks(model):
    return len(model.encoder.layers[0].att.module._forward_hooks)


def test_dict_output_with_tensor_actions_and_per_instance_matrices():
    # Decoder actions start at the first bin (the depot start is implied) and end at the depot.
    pi = torch.tensor([[1, 2, 0], [2, 1, 0]])
    dist = torch.stack([torch.arange(9.0).view(3, 3), 10 * torch.arange(9.0).view(3, 3)])
    agent = NeuralAgent(_Model(pi))
    _, ret, attn = agent.compute_batch_sim({"locs": torch.zeros(2, 2, 2)}, dist)

    def tour_len(d, tour):
        return sum(float(d[a, b]) for a, b in zip(tour[:-1], tour[1:], strict=True))

    expected = torch.tensor([tour_len(dist[0], [0, 1, 2, 0]), tour_len(dist[1], [0, 2, 1, 0])])
    torch.testing.assert_close(ret["km"], expected)
    assert attn["attention_weights"].numel() == 1


def test_shared_matrix_km_is_unchanged():
    pi = torch.tensor([[1, 2, 0], [2, 1, 0]])
    dist = torch.arange(9.0).view(3, 3)
    _, ret, _ = NeuralAgent(_Model(pi)).compute_batch_sim({"locs": torch.zeros(2, 2, 2)}, dist)
    # d[0,1] + d[1,2] + d[2,0] and d[0,2] + d[2,1] + d[1,0]
    torch.testing.assert_close(ret["km"], torch.tensor([1.0 + 5.0 + 6.0, 2.0 + 7.0 + 3.0]))


def test_attention_hooks_are_removed_after_each_call():
    model = _Model(torch.tensor([[1, 0]]))
    agent = NeuralAgent(model)
    for _ in range(3):
        agent.compute_batch_sim({"locs": torch.zeros(1, 1, 2)}, torch.ones(2, 2))
    assert _hooks(model) == 0
