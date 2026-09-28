from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.policies.route_construction.learning_algorithms.neural_agent.policy_na import (
    NeuralAgentPolicy,
)
from logic.src.policies.route_construction.learning_algorithms.neural_agent.simulation import (
    SimulationMixin,
)


class DummyNeuralAgent(SimulationMixin):
    def __init__(self) -> None:
        self.model = MagicMock()
        encoder = nn.Linear(4, 4)
        self.model.encoder = encoder


def test_b_gemini_01_normalization_forwarded_to_gat_encoder() -> None:
    """Verify that normalization='layer' in AttentionModelPolicy creates LayerNorm,

    not BatchNorm1d (B-gemini-01 regression test).
    """
    policy = AttentionModelPolicy(
        env_name="vrpp",
        embed_dim=32,
        n_encode_layers=2,
        normalization="layer",
    )
    # The encoder layers should use LayerNorm, not BatchNorm1d
    first_layer = policy.encoder.layers[0]
    norm1 = first_layer.norm1.normalizer
    norm2 = first_layer.norm2.normalizer
    assert isinstance(norm1, nn.LayerNorm), f"Expected LayerNorm, got {type(norm1)}"
    assert isinstance(norm2, nn.LayerNorm), f"Expected LayerNorm, got {type(norm2)}"

    # When normalization='batch', should use BatchNorm1d
    policy_batch = AttentionModelPolicy(
        env_name="vrpp",
        embed_dim=32,
        n_encode_layers=2,
        normalization="batch",
    )
    norm1_batch = policy_batch.encoder.layers[0].norm1.normalizer
    assert isinstance(norm1_batch, nn.BatchNorm1d), f"Expected BatchNorm1d, got {type(norm1_batch)}"


def test_b_gemini_02_empty_mandatory_returns_depot_loop() -> None:
    """Verify that empty mandatory set returns [0, 0] instead of [0]

    under Owner Ruling D3 (B-gemini-02 regression test).
    """
    agent = DummyNeuralAgent()
    graph = (None, None)
    # mandatory tensor with 5 nodes, all False (empty mandatory set)
    mandatory = torch.zeros(5, dtype=torch.bool)
    result = agent.compute_simulator_day(
        input={},
        graph=graph,
        distC=1.0,
        profit_vars={},
        waste_history=None,
        cost_weights=None,
        mandatory=mandatory,
    )
    route, cost, metadata = result
    assert route == [0, 0], f"Expected empty route [0, 0], got {route}"
    assert cost == 0
    assert metadata.get("mandatory_empty") is True


def test_b_gemini_03_revenue_units_and_ground_truth() -> None:
    """Verify that NeuralAgentPolicy revenue uses real_c (not noisy c)

    and converts percent fill to kg using volume and density (B-gemini-03 regression test).
    """

    class MockBins:
        def __init__(self) -> None:
            self.c = [50.0, 50.0, 50.0]
            self.real_c = [80.0, 60.0, 40.0]
            self.volume = 1000.0  # liters
            self.density = 0.2  # kg / liter

        def get_level_history(self, device=None):
            return None

    policy = NeuralAgentPolicy()
    tour = [0, 1, 2, 0]
    cost = 10.0
    profit_vars = {
        "revenue_kg": 0.5,  # 0.5 € / kg
    }
    bins = MockBins()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "logic.src.policies.route_construction.learning_algorithms.neural_agent.policy_na.NeuralAgent.compute_simulator_day",
            lambda *args, **kwargs: (tour, cost, {}),
        )

        ret_tour, ret_cost, ret_profit, _, _ = policy.execute(
            model_env=MagicMock(),
            model_ls=({"waste": None}, (None, None), profit_vars),
            dm_tensor=torch.zeros((4, 4)),
            fill=[50.0, 50.0, 50.0],
            bins=bins,
            profit_vars=profit_vars,
            device=torch.device("cpu"),
        )

        assert ret_tour == tour
        assert ret_cost == cost
        assert pytest.approx(ret_profit, 1e-4) == 130.0


def test_validate_mandatory_empty_and_nonempty_cases() -> None:
    """Verify _validate_mandatory handles None, lists, tuples, sets, and tensors

    (both 1D/2D boolean masks and integer ID lists/tensors) without ambiguous
    truth value errors (Codex Finding 10.2 #1 regression test).
    """
    policy = NeuralAgentPolicy()

    # None returns None (not an empty constraint)
    assert policy._validate_mandatory(None) is None

    # Lists
    assert policy._validate_mandatory([]) == ([0, 0], 0.0, 0.0)
    assert policy._validate_mandatory([1, 2]) is None

    # Tuples and sets
    assert policy._validate_mandatory(()) == ([0, 0], 0.0, 0.0)
    assert policy._validate_mandatory((1, 2)) is None
    assert policy._validate_mandatory(set()) == ([0, 0], 0.0, 0.0)
    assert policy._validate_mandatory({1, 2}) is None

    # Integer tensors
    assert policy._validate_mandatory(torch.tensor([], dtype=torch.long)) == ([0, 0], 0.0, 0.0)
    assert policy._validate_mandatory(torch.tensor([1, 2], dtype=torch.long)) is None

    # Boolean tensors (1D)
    assert policy._validate_mandatory(torch.tensor([False, False, False])) == ([0, 0], 0.0, 0.0)
    assert policy._validate_mandatory(torch.tensor([False, True, False])) is None

    # Boolean tensors (2D)
    assert policy._validate_mandatory(torch.tensor([[False, False, False]])) == ([0, 0], 0.0, 0.0)
    assert policy._validate_mandatory(torch.tensor([[False, True, False]])) is None

    # Empty 0-element tensor
    assert policy._validate_mandatory(torch.tensor([])) == ([0, 0], 0.0, 0.0)


def test_execute_multi_element_tensor_mask_no_ambiguous_boolean_crash() -> None:
    """Verify that NeuralAgentPolicy.execute does not crash with

    'Boolean value of Tensor with more than one value is ambiguous'
    when passed a multi-element boolean tensor mask (Codex finding 10.2 #1).
    """
    policy = NeuralAgentPolicy()
    mask = torch.tensor([[False, True, False]])

    # Because mask has True at index 1, it is non-empty; validation returns None.
    # Execution proceeds to model unpacking, raising KeyError('model_env'), NOT RuntimeError.
    with pytest.raises(KeyError) as exc_info:
        policy.execute(mandatory=mask)
    assert "model_env" in str(exc_info.value)


def test_execute_empty_tensor_mask_early_exit() -> None:
    """Verify that NeuralAgentPolicy.execute returns early [0, 0] without error

    when passed an empty list or an empty boolean tensor mask.
    """
    policy = NeuralAgentPolicy()

    # Empty list
    res_list = policy.execute(mandatory=[])
    assert res_list[0] == [0, 0]
    assert res_list[1] == 0.0
    assert res_list[2] == 0.0

    # Empty 2D boolean tensor mask
    res_tensor = policy.execute(mandatory=torch.tensor([[False, False, False]]))
    assert res_tensor[0] == [0, 0]
    assert res_tensor[1] == 0.0
    assert res_tensor[2] == 0.0
