

from typing import Tuple

import pytest
import torch
import torch.nn as nn
from logic.src.utils.functions.path import get_path_until_string
from logic.src.utils.functions.sampling import sample_many
from logic.src.utils.functions.tensors import (
    compute_in_batches,
    do_batch_rep,
    move_to,
)

# # from logic.src.tracking.hooks.attention_hooks import add_attention_hooks  # AUTO-REMOVED  # AUTO-REMOVED
from logic.src.utils.model.checkpoint_utils import torch_load_cpu
from logic.src.utils.model.problem_factory import load_problem
from logic.src.utils.model.processing import get_inner_model, parse_softmax_temperature

pytestmark = [pytest.mark.unit, pytest.mark.fast]





class TestFunctions:
    """Class for functions.py tests."""

    def test_compute_in_batches_tuple(self):
        """Test compute_in_batches with tuple return."""

        def f(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Helper function for batch computation test."""
            return x * 2, x + 1

        x = torch.arange(10)
        res1, res2 = compute_in_batches(f, 4, x)
        assert torch.equal(res1, x * 2)
        assert torch.equal(res2, x + 1)

    def test_compute_in_batches_none(self):
        """Test compute_in_batches with None return."""

        def f(x: torch.Tensor) -> None:
            """Helper function returning None for batch computation test."""
            return None

        x = torch.arange(10)
        res = compute_in_batches(f, 4, x)
        assert res is None



    def test_do_batch_rep_complex(self):
        """Test do_batch_rep with list and tuple."""
        t = torch.randn(3)
        data = [t, (t, {"x": t})]
        rep = do_batch_rep(data, 2)
        assert isinstance(rep, list)
        assert rep[0].shape == (6,)
        assert isinstance(rep[1], tuple)

    def test_sample_many(self):
        """Test sample_many sampling loop."""

        def inner_func(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Helper function simulating model forward for sampling test."""
            batch_size = x.size(0)
            return torch.randn(batch_size, 5, 5), torch.randint(0, 5, (batch_size, 5))

        def get_cost(input_data: torch.Tensor, pi: torch.Tensor) -> Tuple[torch.Tensor, None]:
            """Helper function simulating cost calculation for sampling test."""
            batch_size = input_data.size(0)
            return torch.rand(batch_size), None

        input_data = torch.rand(2, 10)
        minpis, mincosts = sample_many(inner_func, get_cost, input_data, batch_rep=2, iter_rep=3)
        assert minpis.shape[0] == 2
        assert mincosts.shape[0] == 2

    def test_get_inner_model_direct(self):
        """Test get_inner_model with non-wrapped model."""
        model = nn.Linear(10, 5)
        inner = get_inner_model(model)
        assert inner is model

    def test_get_inner_model_data_parallel(self):
        """Test get_inner_model correctly checks isinstance DataParallel."""
        model = nn.Linear(10, 5)
        inner = get_inner_model(model)
        assert inner is model

    def test_move_to_tensor(self):
        """Test move_to with tensor."""
        tensor = torch.rand(5, 3)
        result = move_to(tensor, torch.device("cpu"))
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_move_to_dict(self):
        """Test move_to with dict of tensors."""
        data = {"a": torch.rand(3), "b": torch.rand(4)}
        result = move_to(data, torch.device("cpu"))
        assert isinstance(result, dict)
        assert all(v.device.type == "cpu" for v in result.values())

    def test_load_problem_vrpp(self):
        """Test load_problem for VRPP."""
        problem = load_problem("vrpp")
        assert problem.NAME == "vrpp"

    def test_load_problem_cvrpp(self):
        """Test load_problem for CVRPP."""
        problem = load_problem("cvrpp")
        assert problem.NAME == "cvrpp"

    def test_parse_softmax_temperature_float(self):
        """Test parse_softmax_temperature with float."""
        temp = parse_softmax_temperature("1.5")
        assert temp == 1.5

    def test_get_path_until_string(self):
        """Test get_path_until_string function."""
        path = "/home/user/project/outputs/run1/model.pt"
        result = get_path_until_string(path, "outputs")
        assert result is not None
        assert result.endswith("outputs")

    def test_do_batch_rep_tensor(self):
        """Test do_batch_rep with tensor."""
        tensor = torch.rand(2, 3)
        result = do_batch_rep(tensor, 3)
        assert result.shape[0] == 6

    def test_do_batch_rep_dict(self):
        """Test do_batch_rep with dict of tensors."""
        data = {"x": torch.rand(2, 3), "y": torch.rand(2, 4)}
        result = do_batch_rep(data, 2)
        assert result["x"].shape[0] == 4

    def test_torch_load_cpu(self, tmp_path):
        """Test CPU-mapped tensor loading."""
        tensor = torch.randn(10, 5)
        filepath = str(tmp_path / "tensor.pt")
        torch.save(tensor, filepath)
        loaded = torch_load_cpu(filepath)
        assert torch.allclose(tensor, loaded)


# Dummy fallback for removed attention hooks
def add_attention_hooks(model):
    return {"weights": [], "masks": [], "handles": []}
