"""Unit tests for function.py utilities."""

import pytest
import torch
import torch.nn as nn
from logic.src.utils.functions import (
    compute_in_batches,
    do_batch_rep,
    get_path_until_string,
    move_to,
)
from logic.src.utils.model import (
    get_inner_model,
    load_data,
    load_problem,
    parse_softmax_temperature,
    torch_load_cpu,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]



class TestGetInnerModel:
    def test_unwraps_dataparallel(self):
        inner = nn.Linear(10, 10)
        wrapped = nn.DataParallel(inner)
        result = get_inner_model(wrapped)
        assert result is inner

    def test_returns_model_if_not_wrapped(self):
        model = nn.Linear(10, 10)
        result = get_inner_model(model)
        assert result is model


class TestLoadProblem:
    def test_loads_ptp(self):
        env_cls = load_problem("ptp")
        assert env_cls is not None

    def test_invalid_problem_raises(self):
        with pytest.raises(AssertionError):
            load_problem("invalid_problem")


class TestTorchLoadCpu:
    def test_loads_to_cpu(self, tmp_path):
        data = {"weight": torch.randn(3, 3)}
        filepath = str(tmp_path / "model.pt")
        torch.save(data, filepath)

        loaded = torch_load_cpu(filepath)
        assert loaded["weight"].device == torch.device("cpu")


class TestLoadData:
    def test_load_from_path(self, tmp_path):
        data = {"key": "value"}
        filepath = str(tmp_path / "data.pt")
        torch.save(data, filepath)

        result = load_data(filepath, None)
        assert result["key"] == "value"

    def test_returns_empty_if_neither(self):
        result = load_data(None, None)
        assert result == {}


class TestMoveTo:
    def test_moves_tensor(self):
        t = torch.randn(3, 3)
        result = move_to(t, torch.device("cpu"))
        assert result.device == torch.device("cpu")

    def test_moves_dict_recursively(self):
        data = {"a": torch.randn(2, 2), "b": {"c": torch.randn(2, 2)}}
        result = move_to(data, torch.device("cpu"))
        assert result["a"].device == torch.device("cpu")
        assert result["b"]["c"].device == torch.device("cpu")


class TestGetPathUntilString:
    def test_truncates_at_string(self):
        path = "/home/user/WSmart-Route/logic/src/models"
        result = get_path_until_string(path, "WSmart-Route")
        assert result is not None
        assert result.endswith("WSmart-Route")

    def test_returns_none_if_not_found(self):
        path = "/home/user/project/src"
        result = get_path_until_string(path, "NotFound")
        assert result is None


class TestComputeInBatches:
    def test_computes_in_batches(self):
        def f(x):
            return x * 2

        tensor = torch.arange(100)
        result = compute_in_batches(f, 10, tensor)
        assert torch.equal(result, tensor * 2)


class TestDoBatchRep:
    def test_replicates_tensor(self):
        t = torch.randn(2, 3)
        result = do_batch_rep(t, 3)
        assert result.shape == (6, 3)

    def test_replicates_dict(self):
        d = {"a": torch.randn(2, 3)}
        result = do_batch_rep(d, 3)
        assert result["a"].shape == (6, 3)


class TestParseSoftmaxTemperature:
    def test_parses_float(self):
        result = parse_softmax_temperature(1.5)
        assert result == 1.5

    def test_parses_string_float(self):
        result = parse_softmax_temperature("2.0")
        assert result == 2.0
