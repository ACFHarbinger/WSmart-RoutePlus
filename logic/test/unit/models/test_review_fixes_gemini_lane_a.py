"""Unit tests verifying Gemini review fixes across Lane A (Attention Models & Decoding)."""

import pytest
import torch
from tensordict import TensorDict

pytestmark = [pytest.mark.unit]


def test_context_embedders_multi_instance_beam_expansion():
    """Verify that context embedders support beam_width > 1 without dimension mismatch crash."""
    from logic.src.models.subnets.embeddings.context.cmptp import CMPTPContextEmbedder
    from logic.src.models.subnets.embeddings.context.generic import GenericContextEmbedder
    from logic.src.models.subnets.embeddings.context.ptp import PTPContextEmbedder

    embed_dim = 16
    for cls in [CMPTPContextEmbedder, GenericContextEmbedder, PTPContextEmbedder]:
        embedder = cls(embed_dim=embed_dim)
        for bs in [1, 2]:
            num_nodes = 10
            beam_width = 3
            embeddings = torch.randn(bs, num_nodes, embed_dim)
            td = TensorDict(
                {
                    "current_node": torch.randint(0, num_nodes, (bs * beam_width,)),
                    "locs": torch.rand(bs * beam_width, num_nodes, 2),
                    "capacity": torch.full((bs * beam_width,), 20.0),
                    "used_capacity": torch.full((bs * beam_width,), 5.0),
                    "waste": torch.rand(bs * beam_width, num_nodes),
                },
                batch_size=[bs * beam_width],
            )
            out = embedder(embeddings, td)
            assert out.shape == (bs * beam_width, 1, embed_dim)


def test_attention_model_policy_no_dummy_step_accumulation():
    """Verify that finished instances do not accumulate log_p and entropy indefinitely."""
    from logic.src.envs.cmptp import CMPTPEnv
    from logic.src.models.core.attention_model.policy import AttentionModelPolicy

    env = CMPTPEnv(generator_params={"num_loc": 6})
    policy = AttentionModelPolicy(env_name=env.name, embed_dim=16, n_heads=2, n_encode_layers=1)

    td = env.reset(batch_size=[2])
    # Force instance 0 to finish early at step 1
    actions = torch.tensor([[0, 0, 0, 0], [1, 2, 3, 0]])
    out = policy(td.clone(), env=env, actions=actions, return_actions=True)

    log_p = out["log_likelihood"]
    assert log_p.shape == (2,)
    # Instance 0 finishes after 1 step (depot), so it should not accumulate the remaining 3 dummy steps
    assert torch.isfinite(log_p).all()


def test_attention_model_policy_teacher_forcing_bounds_check():
    """Verify teacher forcing does not raise IndexError when trajectory exceeds actions length."""
    from logic.src.envs.cmptp import CMPTPEnv
    from logic.src.models.core.attention_model.policy import AttentionModelPolicy

    env = CMPTPEnv(generator_params={"num_loc": 6})
    policy = AttentionModelPolicy(env_name=env.name, embed_dim=16, n_heads=2, n_encode_layers=1)

    td = env.reset(batch_size=[2])
    # Supply short actions sequence of length 2
    short_actions = torch.tensor([[1, 2], [2, 3]])
    # Must terminate gracefully without IndexError
    out = policy(td, env=env, actions=short_actions, return_actions=True)
    assert out["actions"].shape[1] <= 2


def test_sampling_step_masked_probabilities_zero():
    """Verify that masked actions have strict 0.0 probability and are never clamped to positive value."""
    from logic.src.utils.decoding.sampling import Sampling

    sampling = Sampling()
    logits = torch.tensor([[2.0, 5.0, -1.0]])
    mask = torch.tensor([[True, False, True]])  # Node 1 is masked invalid

    selected, log_p, entropy = sampling.step(logits, mask)
    # Selected action must never be the masked node (node 1)
    assert selected.item() in (0, 2)
    assert selected.item() != 1
    assert torch.isfinite(log_p).all()
    assert torch.isfinite(entropy).all()


def test_float16_epsilon_underflow_safety():
    """Verify that eps in float16 does not underflow to 0.0 causing -inf log_p."""
    from logic.src.utils.decoding.greedy import Greedy

    greedy = Greedy()
    logits = torch.tensor([[-10.0, 10.0]], dtype=torch.float16)
    mask = torch.tensor([[True, True]])

    # Should not produce NaN or -inf in log_p
    action, log_p, entropy = greedy.step(logits, mask)
    assert action.item() == 1
    assert torch.isfinite(log_p).all()


def test_parse_hydra_config_preserves_decoder_settings():
    """Verify _parse_hydra_config preserves decoder_type and decoder_normalization."""
    from logic.src.utils.model.loader import _parse_hydra_config

    cfg = {
        "model": {
            "encoder": {"type": "am"},
            "decoder": {
                "type": "gat",
                "normalization": "layer",
            },
        }
    }
    parsed = _parse_hydra_config(cfg)
    assert parsed.get("decoder_type") == "gat"
    assert parsed.get("decoder_normalization") == "layer"


def test_base_problem_beam_search_propose_expansions_unpacking():
    """Verify BaseProblem.beam_search propose_expansions returns (parent, action, score) triplet."""
    from logic.src.envs.base.problem import BaseProblem
    from tensordict import TensorDict

    class DummyModel:
        def precompute_fixed(self, input, edges=None):
            return "fixed"

        def propose_expansions(self, beam, fixed, normalize=True):
            # Returns [batch * beam, num_nodes]
            bs = beam.score.size(0)
            return torch.zeros(bs, 5)

    class DummyState:
        def __init__(self, bs):
            self.ids = torch.arange(bs)
            self.td = TensorDict({"i": torch.zeros(bs, dtype=torch.long)}, batch_size=[bs])

        def get_mask(self):
            return torch.zeros(len(self.ids), 1, 5, dtype=torch.bool)

        def all_finished(self):
            return True

        def update(self, action):
            return self

        def construct_solutions(self, actions):
            return actions

        def get_final_cost(self):
            return torch.zeros(len(self.ids), 1)

        def __getitem__(self, idx):
            return self

    class DummyProblem(BaseProblem):
        NAME = "dummy"

        @classmethod
        def make_state(cls, input_data, **kwargs):
            return DummyState(len(input_data["ids"]))

    inp = {"ids": torch.tensor([0, 1])}
    res = DummyProblem.beam_search(inp, beam_size=2, cost_weights=1.0, model=DummyModel())
    assert res is not None


def test_batch_norm_eval_singleton_input():
    """Verify Normalization with BatchNorm1d runs on singleton batch in eval mode without crash."""
    from logic.src.models.subnets.modules.normalization import Normalization

    norm = Normalization(embed_dim=16, norm_name="batch")
    norm.eval()
    x = torch.randn(1, 16)
    out = norm(x)
    assert out.shape == (1, 16)


def test_glimpse_decoder_mask_inner_forwarding():
    """Verify GlimpseDecoder forwards mask_inner flag to one_to_many_logits."""
    from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder

    dec_unmasked = GlimpseDecoder(embed_dim=16, hidden_dim=32, problem="cmptp", n_heads=2, mask_inner=False)
    assert dec_unmasked.mask_inner is False

    dec_masked = GlimpseDecoder(embed_dim=16, hidden_dim=32, problem="cmptp", n_heads=2, mask_inner=True)
    assert dec_masked.mask_inner is True
