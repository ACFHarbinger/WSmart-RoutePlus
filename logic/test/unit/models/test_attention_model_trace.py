"""Tests for AttentionModel execution trace instrumentation (Issue #171).

Issue #171:
- one CPU forward of the attention model records encoder and decoder steps on the same schema.
- Masked nodes stay masked.
- Do not trace WeightAdjustmentRNN as a route decoder.
- The default training loop must not write a trace.
"""

from typing import Tuple

import pytest
import torch
from logic.src.envs.cmptp import CMPTPEnv
from logic.src.models.core.attention_model import (
    AttentionModel,
    build_attention_model_trace,
)
from logic.src.models.meta.weight_adjustment_rnn import WeightAdjustmentRNN
from logic.src.policies.catalog import build_catalog
from logic.src.policies.route_improvement.common.helpers import split_tour
from logic.src.policies.trace_runner import routes_equal
from logic.src.tracking.execution_trace import (
    SCHEMA_VERSION,
    ExecutionTrace,
    validate_trace,
)
from tensordict import TensorDict

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _make_cmptp_instance(seed: int = 42, n_nodes: int = 5) -> Tuple[AttentionModel, CMPTPEnv, TensorDict]:
    torch.manual_seed(seed)
    model = AttentionModel(embed_dim=32, hidden_dim=64, problem="cmptp", n_encode_layers=1, n_heads=4)
    model.eval()
    env = CMPTPEnv(generator_params={"num_loc": n_nodes}, check_env_specs=False)
    g = torch.Generator().manual_seed(seed)
    td = TensorDict(
        {
            "locs": torch.rand(1, n_nodes, 2, generator=g),
            "depot": torch.tensor([[0.5, 0.5]]),
            "waste": torch.rand(1, n_nodes, generator=g) * 0.8 + 0.1,
            "capacity": torch.tensor([2.0]),
        },
        batch_size=[1],
    )
    td = env.reset(td)
    return model, env, td


def test_one_cpu_forward_records_encoder_and_decoder_steps():
    """One CPU forward records schema-1 trace with encoder stage and decoder steps."""
    model, env, td = _make_cmptp_instance(seed=42, n_nodes=5)

    out = model(td.clone(), env, strategy="greedy", trace=True, seed=42)
    assert "trace" in out
    trace = out["trace"]
    assert isinstance(trace, ExecutionTrace)

    doc = trace.to_dict()
    validate_trace(doc)

    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["problem"] == "cmptp"
    assert doc["policies"]["model"] == "am"

    kinds = [e["kind"] for e in doc["events"]]
    names = [e["name"] for e in doc["events"]]

    assert "input" in kinds
    assert "encoder" in names
    assert "decoder" in names
    assert "decoder_step" in names
    assert "routes" in names
    assert doc["events"][-1]["kind"] == "output"

    encoder_event = next(e for e in doc["events"] if e["name"] == "encoder")
    assert encoder_event["kind"] == "stage"
    assert encoder_event["arguments"]["embed_dim"] == 32
    assert encoder_event["arguments"]["n_heads"] == 4
    assert encoder_event["arguments"]["n_layers"] == 1

    decoder_stage = next(e for e in doc["events"] if e["name"] == "decoder")
    assert decoder_stage["kind"] == "stage"
    assert decoder_stage["parent_id"] == encoder_event["id"]

    decoder_steps = [e for e in doc["events"] if e["name"] == "decoder_step"]
    assert len(decoder_steps) > 0

    assert doc["final"] is not None
    assert "routes" in doc["final"]
    assert isinstance(doc["final"]["routes"], list)


def test_masked_nodes_stay_masked_across_decoding_steps():
    """Masked nodes remain masked in subsequent steps, and action is never masked."""
    model, env, td = _make_cmptp_instance(seed=42, n_nodes=5)

    out = model(td.clone(), env, strategy="greedy", trace=True, seed=42)
    trace = out["trace"]
    doc = trace.to_dict()
    validate_trace(doc)

    decoder_steps = [e for e in doc["events"] if e["name"] == "decoder_step"]
    assert len(decoder_steps) > 0

    previously_masked = set()
    for step_event in decoder_steps:
        args = step_event["arguments"]
        action = args["action"]
        masked_nodes = set(args["masked_nodes"])
        mask_list = args["mask"]

        # Action selected must NOT be a masked node
        assert action not in masked_nodes, f"Action {action} was chosen from masked nodes {masked_nodes}"

        # Masked nodes stay masked: every node masked in a previous step must still be masked
        for node in previously_masked:
            assert node in masked_nodes, f"Node {node} was previously masked but became unmasked at step {args['step']}"
            assert mask_list[node] is True

        # If a non-depot customer node is visited, it becomes masked in all subsequent steps
        if action != 0:
            previously_masked.add(action)


def test_traced_forward_matches_untraced_forward_exact_seed():
    """Traced and untraced solve on same seed must produce identical actions, reward, and tours."""
    model, env, td = _make_cmptp_instance(seed=42, n_nodes=5)

    out_untraced = model(td.clone(), env, strategy="greedy")
    out_traced = model(td.clone(), env, strategy="greedy", trace=True, seed=42)

    assert torch.equal(out_untraced["actions"], out_traced["actions"])
    assert out_untraced["reward"].item() == pytest.approx(out_traced["reward"].item(), rel=1e-6)

    # Final tours must be identical
    untraced_routes = split_tour(out_untraced["actions"][0].tolist())
    if not untraced_routes:
        untraced_routes = [out_untraced["actions"][0].tolist()]

    traced_routes = out_traced["trace"].final["routes"]
    assert routes_equal(untraced_routes, traced_routes), f"Tours differ: {untraced_routes} != {traced_routes}"


def test_sampling_strategy_seed_parity():
    """Sampling decoding with fixed manual seed matches exactly between untraced and traced."""
    model, env, td = _make_cmptp_instance(seed=123, n_nodes=6)

    torch.manual_seed(999)
    out_untraced = model(td.clone(), env, strategy="sampling")

    torch.manual_seed(999)
    out_traced = model(td.clone(), env, strategy="sampling", trace=True, seed=999)

    assert torch.equal(out_untraced["actions"], out_traced["actions"])
    assert out_untraced["reward"].item() == pytest.approx(out_traced["reward"].item(), rel=1e-6)

    untraced_routes = split_tour(out_untraced["actions"][0].tolist())
    if not untraced_routes:
        untraced_routes = [out_untraced["actions"][0].tolist()]
    assert routes_equal(untraced_routes, out_traced["trace"].final["routes"])


def test_build_attention_model_trace_helper():
    """build_attention_model_trace helper constructs and runs a valid trace."""
    trace, out = build_attention_model_trace(problem="cmptp", graph_size=5, seed=1, strategy="greedy")
    doc = trace.to_dict()
    validate_trace(doc)

    assert doc["schema_version"] == 1
    assert doc["events"][0]["kind"] == "input"
    assert doc["events"][1]["name"] == "encoder"
    assert doc["events"][2]["name"] == "decoder"
    assert doc["final"]["routes"] is not None


def test_custom_execution_trace_passed_in():
    """Passing a pre-constructed ExecutionTrace populates it correctly."""
    model, env, td = _make_cmptp_instance(seed=42, n_nodes=5)

    custom_trace = ExecutionTrace(
        problem="cmptp",
        policies={"model": "am"},
        seed=42,
        instance={"n_nodes": 5, "depot": [0.5, 0.5]},
    )
    root_id = custom_trace.add(
        name="custom_input",
        kind="input",
        arguments={"custom": True},
    )

    out = model(td.clone(), env, strategy="greedy", trace=custom_trace, trace_parent_id=root_id)
    assert out["trace"] is custom_trace
    doc = custom_trace.to_dict()
    validate_trace(doc)

    # First event is custom_input, second is encoder parented to custom_input
    assert doc["events"][0]["name"] == "custom_input"
    assert doc["events"][1]["name"] == "encoder"
    assert doc["events"][1]["parent_id"] == root_id


def test_default_training_loop_does_not_write_trace():
    """Default forward pass passes trace=None and writes no trace."""
    model, env, td = _make_cmptp_instance(seed=42, n_nodes=5)

    out = model(td.clone(), env, strategy="greedy")
    assert "trace" not in out


def test_weight_adjustment_rnn_not_traced_as_route_decoder():
    """WeightAdjustmentRNN is a reward-weight adjustment model, not a route decoder."""
    # Verify module instantiation and forward contract
    rnn = WeightAdjustmentRNN(input_size=4, hidden_size=16, output_size=2, num_layers=1)
    x = torch.randn(2, 5, 4)
    out, hidden = rnn(x)
    assert out.shape == (2, 2)
    assert not hasattr(rnn, "decoder")
    assert not hasattr(rnn, "trace")

    # Verify catalog definition
    catalog = build_catalog()
    models = {m["id"]: m for m in catalog["models"]}
    assert "WeightAdjustmentRNN" in models
    entry = models["WeightAdjustmentRNN"]
    assert entry["role"] == "reward_weight_adjustment"
    assert entry["stages"] == []  # Not a routing model with route decoding stages
    assert entry["produces"] == ["weights"]
