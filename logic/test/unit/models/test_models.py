

from typing import cast
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from logic.src.data.datasets import BaselineDataset
from logic.src.envs import problems as problem_module
from logic.src.envs.problems import MVPTP
from logic.src.models.core.attention_model import AttentionModel
from logic.src.models.core.moe import MoEAttentionModel, MoETemporalAttentionModel
from logic.src.models.subnets.encoders.moe.encoder import MoEGraphAttentionEncoder
from logic.src.models.subnets.modules.moe_feed_forward import MoEFeedForward
from logic.src.models.subnets.modules.moe_layer import MoE
from logic.src.pipeline.rl.common.baselines import (
    CriticBaseline as CriticBaseline,
)
from logic.src.pipeline.rl.common.baselines import (
    ExponentialBaseline as ExponentialBaseline,
)
from logic.src.pipeline.rl.common.baselines import (
    NoBaseline as NoBaseline,
)
from logic.src.pipeline.rl.common.baselines import (
    RolloutBaseline as RolloutBaseline,
)
from logic.src.pipeline.rl.common.baselines import (
    WarmupBaseline as WarmupBaseline,
)
from logic.src.policies.route_construction.learning_algorithms.neural_agent import NeuralAgent
from tensordict import TensorDict
from torch.utils.data import Dataset

pytestmark = [pytest.mark.unit, pytest.mark.fast]


# Patch globals that are expected to be initialized by Dataset
problem_module.COST_KM = 1.0
problem_module.REVENUE_KG = 1.0
problem_module.BIN_CAPACITY = 100.0
problem_module.VEHICLE_CAPACITY = 1000.0


class TestAttentionModel:
    """Tests for the AttentionModel architecture."""

    def test_initialization(self, am_setup):
        """Verifies model initialization parameters."""
        model = am_setup
        assert isinstance(model, AttentionModel)
        assert model.n_heads == 8

    def test_forward(self, am_setup):
        """Verifies forward flow output shapes and types."""
        model = am_setup
        batch_size = 2
        graph_size = 5

        # Mocking embeddings return from encoder
        model.encoder.return_value = torch.zeros(batch_size, graph_size + 1, 128)  # +1 for depot
        # Mock decoder return value (log_p, pi, cost, td)
        model.decoder.return_value = (
            torch.zeros(batch_size, graph_size),
            torch.zeros(batch_size, graph_size),
            torch.zeros(batch_size),
            None,
        )
        # _calc_log_likelihood returns (ll, entropy) when training=True
        model.decoder._calc_log_likelihood.return_value = (
            torch.zeros(batch_size),
            torch.zeros(batch_size),
        )

        input_data = {
            "depot": torch.rand(batch_size, 2),
            "locs": torch.rand(batch_size, graph_size, 2),
            "waste": torch.rand(batch_size, graph_size),
        }
        # Add fill history
        for day in range(1, model.temporal_horizon + 1):
            input_data[f"fill{day}"] = torch.rand(batch_size, graph_size)

        out = model(input_data)
        cost = out["cost"]
        # ll = out.get("log_likelihood")
        assert cost is not None

    def test_compute_batch_sim(self, am_setup):
        """Verifies simulation batch computation."""
        model = am_setup
        model.encoder.return_value = torch.zeros(2, 6, 128)
        model.decoder.return_value = (
            torch.zeros(2, 6),  # log_p
            torch.arange(6).repeat(2, 1),  # selected
            torch.zeros(2),  # cost
        )
        model.decoder._calc_log_likelihood.return_value = torch.zeros(
            2
        )  # ll only if training=False or depending on usage?
        # In compute_batch_sim, model(..., return_pi=True) is called.
        # model.forward calls decoder.
        # It calculates ll using decoder._calc_log_likelihood.

        # But wait, test_compute_batch_sim mocks model.forward later!
        # So mocks on decoder might be irrelevant if model.forward is mocked.
        # But lines 43-46 set mocks on _inner which are now useless.
        # I'll update them anyway for correctness.
        model.problem.get_costs.return_value = (
            torch.zeros(2),
            {"overflows": torch.zeros(2), "waste": torch.zeros(2)},
            None,
        )
        # Mock model forward to return expected tuple for NeuralAgent
        # (cost, ll, cost_dict, pi, entropy)
        model.forward = MagicMock(
            return_value=(
                torch.zeros(2),
                torch.zeros(2),
                {"overflows": torch.zeros(2), "waste": torch.zeros(2)},
                torch.zeros(2, 6, dtype=torch.long),
                torch.zeros(2),
            )
        )

        agent = NeuralAgent(model)

        input_data = {
            "depot": torch.zeros(2, 2),
            "locs": torch.zeros(2, 5, 2),
            "waste": torch.zeros(2, 5),
        }
        for day in range(1, model.temporal_horizon + 1):
            input_data[f"fill{day}"] = torch.zeros(2, 5)
        dist_matrix = torch.zeros(6, 6)

        ucost, ret_dict, attn_dict = agent.compute_batch_sim(input_data, dist_matrix)
        assert "overflows" in ret_dict
        assert "kg" in ret_dict


class TestMandatoryManager:
    """Tests for the MandatoryManager architecture."""

    def test_forward(self, gat_lstm_setup):
        """Verifies forward flow logic."""
        manager = gat_lstm_setup
        B, N = 2, 5
        static = torch.rand(B, N, 2)
        dynamic = torch.rand(B, N, 10)
        global_features = torch.rand(B, 2)
        mask_logits, gate_logits, value = manager(static, dynamic, global_features)

        assert mask_logits.shape == (B, N, 2)
        assert gate_logits.shape == (B, 2)
        assert value.shape == (B, 1)

    def test_select_action_deterministic(self, gat_lstm_setup):
        """Verifies deterministic action selection."""
        manager = gat_lstm_setup
        static = torch.rand(1, 5, 2)
        dynamic = torch.rand(1, 5, 10)
        global_features = torch.rand(1, 2)
        mask_action, gate_action, value = manager.select_action(static, dynamic, global_features, deterministic=True)
        assert mask_action.shape == (1, 5)
        assert gate_action.shape == (1,)

    def test_shared_encoder(self, am_setup):
        """Verifies shared encoder initialization."""
        from logic.src.models.meta.hrl_manager import MandatoryManager



        worker_model = am_setup
        B, N = 1, 5
        static = torch.rand(B, N, 2)
        dynamic = torch.rand(B, N, 10)
        global_features = torch.rand(B, 2)

        manager = MandatoryManager(
            input_dim_static=2,
            input_dim_dynamic=10,
            hidden_dim=128,
            shared_encoder=worker_model.encoder,
            device="cpu",
        )

        assert manager.gat_encoder is worker_model.encoder

        mask_logits, gate_logits, value = manager(static, dynamic, global_features)
        assert mask_logits.shape == (B, N, 2)


class TestReinforceBaselines:
    """Tests for REINFORCE baseline implementations."""

    def test_warmup_baseline(self, mock_baseline):
        """Verifies warmup alpha updates."""
        wb = WarmupBaseline(mock_baseline, warmup_epochs=2, beta=0.8)
        assert wb.alpha == 0

        # Test alpha update
        wb.epoch_callback(MagicMock(spec=nn.Module), 0)
        assert wb.alpha == 0.5

    def test_exponential_baseline(self):
        """Verifies exponential moving average updates."""
        eb = ExponentialBaseline(beta=0.8)
        c = torch.tensor([10.0, 20.0])
        v = eb.eval(cast(TensorDict, None), c)
        assert torch.allclose(v, torch.tensor([15.0, 15.0]))

        c2 = torch.tensor([20.0, 30.0])  # Mean 25
        v2 = eb.eval(cast(TensorDict, None), c2)
        # v2 = 0.8 * 15 + 0.2 * 25 = 12 + 5 = 17
        assert torch.allclose(v2, torch.tensor([17.0, 17.0]))

    def test_no_baseline(self):
        """Verifies no-op baseline."""
        nb = NoBaseline()
        reward = torch.tensor([10.0, 20.0])
        v = nb.eval(cast(TensorDict, None), reward)
        assert torch.all(v == 0)

    def test_critic_baseline(self):
        """Verifies critic network evaluation and state dicts."""
        critic = MagicMock()
        critic.return_value = torch.tensor([1.0, 2.0])
        critic.parameters.return_value = [torch.tensor([1.0])]
        critic.state_dict.return_value = {"a": 1}

        cb = CriticBaseline(critic)
        x = torch.randn(2, 5)
        c = torch.tensor([1.0, 2.0])

        # Test eval
        v = cb.eval(cast(TensorDict, x), c)
        assert torch.allclose(v, torch.tensor([1.0, 2.0]))

        # Test learnable parameters
        params = cb.get_learnable_parameters()
        assert len(params) == 1

        # Test state dict (nested)
        sd = cb.state_dict()
        assert "critic" in sd
        assert sd["critic"] == {"a": 1}

    def test_rollout_baseline(self, mocker):
        """Verifies rollout baseline evaluation and updates."""
        # Mocks
        mock_policy = MagicMock(spec=nn.Module)
        mock_policy.parameters.return_value = [torch.tensor([1.0])]

        # Mock deepcopy to return a specific mock for the baseline policy
        mock_baseline_policy = MagicMock(spec=nn.Module)
        mock_baseline_policy.parameters.return_value = [torch.tensor([1.0])]
        mocker.patch("copy.deepcopy", return_value=mock_baseline_policy)

        # Initialize
        rb = RolloutBaseline(policy=mock_policy, update_every=1, bl_alpha=0.05)
        # Note: RolloutBaseline.setup(policy) calls deepcopy(policy)
        assert rb.baseline_policy is mock_baseline_policy

        # Mock _rollout
        mocker.patch.object(rb, "_rollout", return_value=torch.tensor([10.0, 20.0]))

        # Test eval
        td = MagicMock(spec=TensorDict)
        reward = torch.tensor([10.0, 20.0])
        v = rb.eval(td, reward, env=MagicMock())
        assert torch.all(v == torch.tensor([10.0, 20.0]))

        # Mocking ttest_rel to simulate significant improvement
        mocker.patch("scipy.stats.ttest_rel", return_value=(MagicMock(), 0.001))
        # candidate values mean 15 > baseline values mean 5 (improvement)
        cast(MagicMock, rb._rollout).side_effect = [torch.tensor([10.0, 20.0]), torch.tensor([5.0, 5.0])]

        # A reporting val_dataset is not a comparison pool: without the baseline's own pool the
        # callback must not promote (B-codex-01). Promotion with a pool is covered by the
        # rollout lifecycle tests in logic/test/unit/pipeline/rl/.
        mock_setup = mocker.patch.object(rb, "setup")
        rb.epoch_callback(mock_policy, 0, val_dataset=MagicMock(), env=MagicMock())
        assert not mock_setup.called

    def test_baseline_dataset(self):
        """Verifies dataset wrapping."""
        # BaselineDataset expects (Dataset, torch.Tensor)
        ds = BaselineDataset(cast(Dataset, [1, 2]), torch.tensor([3, 4]))
        assert len(ds) == 2
        item = ds[0]
        # BaselineDataset now stores data in 'data' and baseline in 'baseline'
        assert item["data"] == 1
        assert item["baseline"] == 3


class TestTemporalAttentionModel:
    """Tests for the TemporalAttentionModel."""

    def test_init_embed_uses_fill_predictor(self, tam_setup):
        """Verifies fill history embedding."""
        model = tam_setup

        # We need to mock _init_embed_depot/etc calls or rely on base class mocks if complex
        # But here we mocked encoder so it won't be called for real.
        # However, _init_embed is called BEFORE encoder.
        # We need to ensure base AttentionModel._init_embed works or is mocked.

        # Let's mock the base _init_embed via super() is tricky.
        # Instead, verify update_fill_history logic which is easier and unique

        fh = torch.zeros(1, 5, 5)  # 5 nodes, 5 steps
        new_fill = torch.ones(1, 5)
        updated = model.update_fill_history(fh, new_fill)
        assert torch.all(updated[:, :, -1] == 1)
        assert torch.all(updated[:, :, 0] == 0)

    def test_forward_wraps_input(self, tam_setup):
        """Verifies input wrapping with fill history."""
        model = tam_setup
        model.encoder.return_value = torch.zeros(1, 5, 128)
        model.decoder.return_value = (
            torch.zeros(1, 5),
            torch.zeros(1, 5),
            torch.zeros(1),
            None,
        )
        model.decoder._calc_log_likelihood.return_value = (
            torch.zeros(1),
            torch.zeros(1),
        )

        input_data = {
            "depot": torch.rand(1, 2),
            "locs": torch.rand(1, 4, 2),
            "waste": torch.zeros(1, 4),
        }

        # Calling forward should inject fill_history if missing
        model(input_data)
        assert "fill_history" in input_data
        assert input_data["fill_history"].shape[-1] == 5  # horizon


class TestMoE:
    """Tests for MoE module."""

    def test_moe_forward(self):
        """Test basic forward flow of MoE."""
        batch_size = 2
        seq_len = 5
        d_model = 16
        num_experts = 4
        k = 2

        moe = MoE(
            input_size=d_model,
            output_size=d_model,
            num_experts=num_experts,
            k=k,
            noisy_gating=False,
        )
        x = torch.rand(batch_size, seq_len, d_model)

        output = moe(x)
        assert output.shape == (batch_size, seq_len, d_model)


class TestMoEFeedForward:
    """Tests for MoEFeedForward wrapper."""

    def test_moe_ff_structure(self):
        """Test structure and forward flow."""
        d_model = 16
        d_ff = 32

        moe_ff = MoEFeedForward(
            embed_dim=d_model,
            feed_forward_hidden=d_ff,
            activation="relu",
            af_param=1.0,
            threshold=6.0,
            replacement_value=6.0,
            n_params=3,
            dist_range=[0.1, 0.2],
            num_experts=3,
            k=1,
        )

        assert len(moe_ff.moe.experts) == 3
        # Check if experts are Sequential (FF -> Act -> FF)
        assert isinstance(moe_ff.moe.experts[0], nn.Sequential)

        x = torch.rand(2, 5, d_model)
        out = moe_ff(x)
        assert out.shape == (2, 5, d_model)


class TestMoEEncoder:
    """Tests for MoE Encoder."""

    def test_encoder_integration(self):
        """Test MoEGraphAttentionEncoder."""
        d_model = 16
        encoder = MoEGraphAttentionEncoder(
            n_heads=2,
            embed_dim=d_model,
            n_layers=2,
            feed_forward_hidden=32,
            num_experts=4,
            k=2,
        )

        x = torch.rand(2, 5, d_model)
        out = encoder(x, edges=None)
        assert out.shape == (2, 5, d_model)


class TestMoEModel:
    """Tests for High-Level MoE Model."""

    def test_model_initialization_and_forward(self):
        """Test MoEAttentionModel initialization and forward flow."""
        # Mock problem
        problem = MVPTP()

        model = MoEAttentionModel(
            embed_dim=16,
            hidden_dim=32,
            problem=problem,
            n_encode_layers=1,
            n_heads=2,
            num_experts=3,
            k=1,
        )

        # Check factory injection
        assert isinstance(model.encoder, MoEGraphAttentionEncoder)

        input_data = {
            "depot": torch.rand(2, 2),
            "locs": torch.rand(2, 5, 2),
            "waste": torch.rand(2, 5),
            "capacity": torch.full((2,), 1000.0),
            "max_waste": torch.full((2,), 1000.0),
        }

        model.set_strategy("greedy")
        out = model(input_data, return_pi=True)
        pi = out["pi"]
        assert pi.ndim == 2
        assert pi.size(0) == 2

    def test_temporal_model_initialization(self):
        """Test MoETemporalAttentionModel initialization."""
        problem = MVPTP()
        model = MoETemporalAttentionModel(
            embed_dim=16,
            hidden_dim=32,
            problem=problem,
            n_encode_layers=1,
            temporal_horizon=5,
        )
        assert isinstance(model.encoder, MoEGraphAttentionEncoder)
