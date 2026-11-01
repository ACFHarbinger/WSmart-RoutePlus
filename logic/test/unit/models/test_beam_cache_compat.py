"""Beam ancestry controls cache rows, including unequal/reordered branches."""

from types import SimpleNamespace

import pytest
import torch
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.envs.cmptp import CMPTPEnv
from logic.src.models.core.attention_model.decoding import DecodingMixin
from logic.src.models.subnets.decoders.gat.decoder import DeepGATDecoder
from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder
from logic.src.utils.data.td_state_wrapper import TensorDictStateWrapper
from logic.src.utils.decoding.batch_beam import BatchBeam


@pytest.mark.parametrize("kind", ["glimpse", "gat"])
def test_real_decoder_cache_follows_reordered_unequal_beams(kind):
    torch.manual_seed(13)
    decoder = (
        GlimpseDecoder(embed_dim=16, hidden_dim=32, problem="cmptp", n_heads=2)
        if kind == "glimpse"
        else DeepGATDecoder(
            embed_dim=16, hidden_dim=32, n_heads=2, n_layers=1, norm_config=NormalizationConfig(norm_type="layer")
        )
    )
    decoder.eval()
    env = CMPTPEnv(generator_params={"num_loc": 4})
    state = TensorDictStateWrapper(env.reset(batch_size=[2]), "cmptp", env)
    cache = decoder._precompute(torch.randn(2, 5, 16))
    model = SimpleNamespace(decoder=decoder)
    # Unequal widths and reordering are valid after expansion/pruning.
    for indices in [torch.tensor([1, 0, 1, 0, 1]), torch.tensor([1, 0])]:
        selected = state[indices]
        torch.testing.assert_close(selected.ids.reshape(-1), indices)
        beam = SimpleNamespace(state=selected)
        actual = DecodingMixin.propose_expansions(model, beam, cache, normalize=True)
        expected = []
        for row, original_id in enumerate(indices):
            lp, _ = decoder._get_log_p(cache[int(original_id)], selected[row : row + 1], normalize=True)
            expected.append(lp.squeeze(1) if lp.dim() == 3 else lp)
        torch.testing.assert_close(actual, torch.cat(expected))


def test_real_environment_preserves_ids_and_prunes_by_original_instance():
    env = CMPTPEnv(generator_params={"num_loc": 4})
    state = TensorDictStateWrapper(env.reset(batch_size=[2]), "cmptp", env)
    beam = BatchBeam.initialize(state)
    parent = torch.tensor([0, 0, 0, 0, 1, 1])
    beam = beam.expand(parent, torch.tensor([1, 2, 3, 4, 1, 2]), torch.tensor([1.0, 4.0, 3.0, 2.0, 6.0, 5.0]))
    torch.testing.assert_close(beam.ids, parent)
    beam = beam.topk(3)
    assert torch.bincount(beam.ids, minlength=2).tolist() == [3, 2]
    # Repeated real env steps through completion retain original ancestry.
    ids = beam.ids.clone()
    for _ in range(2):
        beam = beam._replace(state=beam.state.update(torch.zeros(5, dtype=torch.long)))
        torch.testing.assert_close(beam.ids, ids)
    assert beam.all_finished()


@pytest.mark.parametrize("kind", ["glimpse", "gat"])
def test_beam_loop_completes_heterogeneous_instances(kind):
    from logic.src.utils.decoding.beam_search import _beam_search

    torch.manual_seed(2)
    decoder = (
        GlimpseDecoder(embed_dim=16, hidden_dim=32, problem="cmptp", n_heads=2)
        if kind == "glimpse"
        else DeepGATDecoder(
            embed_dim=16, hidden_dim=32, n_heads=2, n_layers=1, norm_config=NormalizationConfig(norm_type="layer")
        )
    )
    decoder.eval()
    env = CMPTPEnv(generator_params={"num_loc": 4})
    td = env.reset(batch_size=[2])
    # One instance can only stop initially; the other has all five choices.
    td["action_mask"][0, 1:] = False
    state = TensorDictStateWrapper(td, "cmptp", env)
    model = SimpleNamespace(decoder=decoder)
    cache = decoder._precompute(torch.randn(2, 5, 16))

    def propose(beam):
        log_p = DecodingMixin.propose_expansions(model, beam, cache, normalize=True)
        scores = log_p + beam.score[:, None]
        parent, action = torch.where(torch.isfinite(scores))
        return parent, action, scores[parent, action]

    beams, final_state = _beam_search(state, 3, propose, keep_states=True)
    assert final_state.all_finished()
    assert len(beams) <= 12
    for beam in beams:
        assert torch.bincount(beam.ids, minlength=2).max().item() <= 3
    assert set(final_state.ids.reshape(-1).tolist()) == {0, 1}


def test_public_cmptp_beam_search_matches_individual_instances():
    from logic.src.envs.cmptp import CMPTP
    from logic.src.models.core.attention_model.model import AttentionModel

    torch.manual_seed(5)
    env = CMPTPEnv(generator_params={"num_loc": 3})
    td = env.generator(batch_size=[2])
    model = AttentionModel(
        embed_dim=16, hidden_dim=32, problem=CMPTP, n_heads=2, n_encode_layers=1, normalization="layer"
    )
    model.eval()
    with torch.no_grad():
        scores, tours, costs, ids, batch_size = model.beam_search(td, beam_size=3, cost_weights=1.0)
        assert batch_size == 2
        # Independently check the returned cost against geometry and unique
        # collections, rather than comparing two calls to the same implementation.
        locs = torch.cat([td["depot"][:, None], td["locs"]], dim=1)
        waste = torch.cat([torch.zeros_like(td["waste"][:, :1]), td["waste"]], dim=1)
        for row, route in enumerate(tours):
            cost, original_id = costs[row], ids[row]
            depot = torch.zeros(1, dtype=route.dtype, device=route.device)
            closed_route = torch.cat([depot, route, depot])
            points = locs[original_id, closed_route]
            distance = (points[1:] - points[:-1]).norm(dim=-1).sum()
            collected = waste[original_id, route.unique()].sum()
            torch.testing.assert_close(cost, distance - collected)
        for index in range(2):
            solo = model.beam_search(td[index : index + 1], beam_size=3, cost_weights=1.0)
            torch.testing.assert_close(scores[ids == index], solo[0])
            torch.testing.assert_close(tours[ids == index], solo[1])
            torch.testing.assert_close(costs[ids == index], solo[2])


@pytest.mark.parametrize("reward_weight", [0.0, 1.0])
def test_simulator_beam_width_larger_than_feasible_candidates(reward_weight):
    from logic.src.configs.policies.routing.learning_algorithms.na import NeuralAgentConfig
    from logic.src.envs.cmptp import CMPTP
    from logic.src.models.core.attention_model.model import AttentionModel
    from logic.src.policies.routing.learning_algorithms.neural_agent.agent import NeuralAgent

    model = AttentionModel(
        embed_dim=16, hidden_dim=32, problem=CMPTP, n_heads=2, n_encode_layers=1, normalization="layer"
    )
    model.eval()
    inputs = {"locs": torch.tensor([[[1.0, 0.0]]]), "depot": torch.zeros(1, 2), "waste": torch.tensor([[0.5]])}
    with torch.no_grad():
        route, cost, _ = NeuralAgent(model).compute_simulator_day(
            inputs,
            (None, torch.zeros(2, 2)),
            torch.zeros(2, 2),
            {"cost_km": 1.0, "revenue_kg": 0.5837, "bin_capacity": 47.5, "vehicle_capacity": 10.0},
            params=NeuralAgentConfig(decoding_strategy="beam_search", beam_width=3, reward_weight=reward_weight),
        )
    assert route[0] == 0
    assert route[-1] == 0
    assert set(route) <= {0, 1}
    assert cost == 0.0
