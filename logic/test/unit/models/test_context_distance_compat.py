"""Checkpoint distance features stay in coordinate units."""

import pytest
import torch
from logic.src.models.subnets.embeddings.context.cptp import CPTPContextEmbedder
from tensordict import TensorDict


@pytest.mark.parametrize("matrix_key", ["dm", "dist", "dist_matrix"])
def test_context_distance_ignores_objective_road_matrix(matrix_key):
    model = CPTPContextEmbedder(4)
    with torch.no_grad():
        model.project_step_context.weight.zero_()
        model.project_step_context.bias.zero_()
        model.project_step_context.weight[0, 5] = 1
    td = TensorDict(
        {
            "current_node": torch.tensor([0]),
            "locs": torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]),
            "waste": torch.tensor([[0.0, 1.0, 1.0]]),
            "visited": torch.tensor([[True, False, False]]),
            "remaining_capacity": torch.tensor([10.0]),
        },
        batch_size=[1],
    )
    embeddings = torch.zeros(1, 3, 4)
    expected = model._step_context(embeddings, td)
    td[matrix_key] = torch.tensor([[[0.0, 10.0, 20.0], [10.0, 0.0, 30.0], [20.0, 30.0, 0.0]]])
    actual = model._step_context(embeddings, td)
    torch.testing.assert_close(actual, expected)
    assert actual[0, 0, 0].item() == 1.0
