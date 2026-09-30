"""Regression checks for ambiguous depot values in the delivered patch."""

import pytest
import torch
from logic.src.envs.tasks.base import BaseProblem


def test_zero_first_customer_without_schema_stays_customer():
    waste = torch.tensor([[0.0, 10.0, 20.0]])
    actual = BaseProblem.get_waste_with_depot({"waste": waste}, torch.tensor([[0, 2, 0]]))
    assert torch.equal(actual, torch.tensor([[0.0, 0.0, 10.0, 20.0]]))


def test_explicit_customer_count_overrides_colocated_customer():
    data = {
        "depot": torch.zeros(1, 2),
        "locs": torch.tensor([[[0.0, 0.0], [2.0, 0.0]]]),
        "waste": torch.tensor([[10.0, 20.0]]),
        "num_loc": 2,
    }
    assert torch.equal(BaseProblem.get_waste_with_depot(data), torch.tensor([[0.0, 10.0, 20.0]]))
    assert BaseProblem.get_tour_length(data, torch.tensor([[0, 2, 0]])).item() == 4.0


def test_invalid_waste_width_is_rejected():
    with pytest.raises(ValueError, match="waste"):
        BaseProblem.get_waste_with_depot({"waste": torch.ones(1, 5), "num_loc": 2})
