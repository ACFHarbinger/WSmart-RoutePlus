"""Execution trace instrumentation for AttentionModel.

Issue #171: one CPU forward of the attention model records encoder and decoder
steps on the schema-1 ExecutionTrace. Masked nodes stay masked.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from logic.src.policies.route_improvement.common.helpers import split_tour
from logic.src.tracking.execution_trace import ExecutionTrace, solution_record


def init_model_trace(
    policy: Any,
    td: Any,
    init_embeds: torch.Tensor,
    trace: Union[bool, ExecutionTrace],
    trace_parent_id: Optional[str] = None,
    strategy: str = "greedy",
    **kwargs: Any,
) -> Tuple[ExecutionTrace, str]:
    """Initialize or bind ExecutionTrace for an AttentionModel forward pass.

    Args:
        policy: The AttentionModel or AttentionModelPolicy instance.
        td: Input TensorDict.
        init_embeds: Contextual node embeddings.
        trace: Either True to allocate a new trace or an existing ExecutionTrace.
        trace_parent_id: Optional parent event ID to anchor under.
        strategy: Decoding strategy identifier.
        kwargs: Additional arguments such as seed.

    Returns:
        Tuple[ExecutionTrace, str]: (trace_object, decoder_stage_id).
    """
    if isinstance(trace, ExecutionTrace):
        trace_obj = trace
    else:
        n_nodes = int(init_embeds.size(1)) if hasattr(init_embeds, "size") and init_embeds.dim() > 1 else 0
        depot_coord = None
        if hasattr(td, "keys") and "depot" in td.keys():
            dep = td["depot"]
            if isinstance(dep, torch.Tensor) and dep.numel() >= 2:
                dep_cpu = dep.cpu()
                depot_coord = [float(dep_cpu[0, 0].item()), float(dep_cpu[0, 1].item())]
        problem_name = str(getattr(policy, "env_name", None) or "cmptp")
        trace_obj = ExecutionTrace(
            problem=problem_name,
            policies={"model": "am"},
            seed=kwargs.get("seed"),
            instance={"n_nodes": n_nodes, "depot": depot_coord},
        )

    if not trace_obj.events:
        batch_size = int(td.batch_size[0]) if hasattr(td, "batch_size") and len(td.batch_size) > 0 else 1
        n_nodes = int(init_embeds.size(1)) if hasattr(init_embeds, "size") and init_embeds.dim() > 1 else 0
        input_id = trace_obj.add(
            name="instance",
            kind="input",
            arguments={"n_nodes": n_nodes, "batch_size": batch_size},
        )
    else:
        input_id = trace_parent_id or trace_obj.events[-1]["id"]

    encoder = getattr(policy, "encoder", None)
    n_layers = getattr(encoder, "n_layers", getattr(policy, "n_encode_layers", 3))
    embed_dim = getattr(policy, "embed_dim", int(init_embeds.size(-1)))
    n_heads = getattr(policy, "n_heads", 8)

    encoder_id = trace_obj.add(
        name="encoder",
        kind="stage",
        parent_id=input_id,
        changed_solution=False,
        arguments={
            "embed_dim": int(embed_dim),
            "n_heads": int(n_heads),
            "n_layers": int(n_layers),
        },
    )

    decoder_stage_id = trace_obj.add(
        name="decoder",
        kind="stage",
        parent_id=encoder_id,
        changed_solution=False,
        arguments={"strategy": str(strategy)},
    )

    return trace_obj, decoder_stage_id


def record_decoder_step(
    trace: ExecutionTrace,
    parent_id: str,
    step_idx: int,
    action: torch.Tensor,
    mask: torch.Tensor,
    log_p: Any,
    current_actions: List[Any],
) -> str:
    """Record an autoregressive decoding step on the ExecutionTrace.

    Args:
        trace: The active ExecutionTrace.
        parent_id: Parent event ID (decoder stage or previous step).
        step_idx: Current 0-based decoding step.
        action: Selected action tensor.
        mask: Invalid nodes mask (True=masked/invalid).
        log_p: Log probability of the selection.
        current_actions: Actions chosen prior to this step.

    Returns:
        str: The newly created event ID.
    """
    chosen_node = int(action[0].item())
    mask_slice = mask[0] if mask.dim() >= 2 else mask
    step_mask_list = [bool(m) for m in mask_slice.tolist()]
    masked_nodes = [i for i, is_masked in enumerate(step_mask_list) if is_masked]

    prior_actions = [int(a[0].item() if isinstance(a, torch.Tensor) else a) for a in current_actions]
    routes_before = split_tour(prior_actions)
    if not routes_before:
        routes_before = [prior_actions] if prior_actions else [[]]

    new_actions = prior_actions + [chosen_node]
    routes_after = split_tour(new_actions)
    if not routes_after:
        routes_after = [[chosen_node]]

    changed = bool(routes_after != routes_before)
    step_log_p = float(log_p[0].item()) if isinstance(log_p, torch.Tensor) else float(log_p)

    step_id = trace.add(
        name="decoder_step",
        kind="operator",
        parent_id=parent_id,
        changed_solution=changed,
        arguments={
            "step": int(step_idx),
            "action": chosen_node,
            "mask": step_mask_list,
            "masked_nodes": masked_nodes,
            "log_p": step_log_p,
        },
        before=solution_record(routes_before),
        after=solution_record(routes_after),
    )
    return step_id


def finalize_model_trace(
    trace: ExecutionTrace,
    parent_id: str,
    output_actions: List[Any],
    reward: Any,
) -> None:
    """Record final routes and set final solution on the trace.

    Args:
        trace: The active ExecutionTrace.
        parent_id: Parent event ID (last step event).
        output_actions: Full list of chosen actions.
        reward: Environment reward tensor or float.
    """
    all_actions = [int(a[0].item() if isinstance(a, torch.Tensor) else a) for a in output_actions]
    final_routes = split_tour(all_actions)
    if not final_routes:
        final_routes = [all_actions] if all_actions else [[]]

    obj = float(reward[0].item()) if isinstance(reward, torch.Tensor) else float(reward)
    final_rec = solution_record(final_routes, objective=obj)

    trace.add(
        name="routes",
        kind="output",
        parent_id=parent_id,
        changed_solution=True,
        after=final_rec,
    )
    trace.set_final(final_routes, objective=obj)


class _NoOpTracer:
    """No-op tracer for default untraced forward passes."""

    def record_step(
        self,
        step_idx: int,
        action: torch.Tensor,
        mask: torch.Tensor,
        log_p: Any,
        current_actions: List[Any],
    ) -> None:
        pass

    def finalize(
        self,
        output_actions: List[Any],
        reward: Any,
        out: Dict[str, Any],
    ) -> None:
        pass


class _ActiveTracer:
    """Active tracer recording events to an ExecutionTrace."""

    def __init__(self, trace_obj: ExecutionTrace, step_parent_id: str) -> None:
        self.trace_obj = trace_obj
        self.step_parent_id: str = step_parent_id

    def record_step(
        self,
        step_idx: int,
        action: torch.Tensor,
        mask: torch.Tensor,
        log_p: Any,
        current_actions: List[Any],
    ) -> None:
        self.step_parent_id = record_decoder_step(
            trace=self.trace_obj,
            parent_id=self.step_parent_id,
            step_idx=step_idx,
            action=action,
            mask=mask,
            log_p=log_p,
            current_actions=current_actions,
        )

    def finalize(
        self,
        output_actions: List[Any],
        reward: Any,
        out: Dict[str, Any],
    ) -> None:
        finalize_model_trace(
            trace=self.trace_obj,
            parent_id=self.step_parent_id,
            output_actions=output_actions,
            reward=reward,
        )
        out["trace"] = self.trace_obj


_NOOP_TRACER = _NoOpTracer()


def make_tracer(
    policy: Any,
    td: Any,
    init_embeds: torch.Tensor,
    strategy: str,
    kwargs: Dict[str, Any],
) -> Any:
    """Construct an active tracer if requested, otherwise return no-op tracer."""
    trace = kwargs.get("trace")
    if trace is None or trace is False:
        return _NOOP_TRACER
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in ("trace", "trace_parent_id")}
    trace_obj, step_parent_id = init_model_trace(
        policy=policy,
        td=td,
        init_embeds=init_embeds,
        trace=trace,
        trace_parent_id=kwargs.get("trace_parent_id"),
        strategy=strategy,
        **extra_kwargs,
    )
    return _ActiveTracer(trace_obj, step_parent_id)


def build_attention_model_trace(
    problem: str = "cmptp",
    graph_size: int = 5,
    seed: int = 1,
    strategy: str = "greedy",
    model: Optional[Any] = None,
    env: Optional[Any] = None,
    td: Optional[Any] = None,
) -> Tuple[ExecutionTrace, Dict[str, Any]]:
    """Build and run one CPU forward of AttentionModel and return (trace, out).

    Args:
        problem: Problem domain ('cmptp', 'ptp', etc.).
        graph_size: Number of nodes.
        seed: Random seed for initialization.
        strategy: Decoding strategy ('greedy', 'sampling').
        model: Optional pre-constructed model.
        env: Optional pre-constructed environment.
        td: Optional pre-constructed input TensorDict.

    Returns:
        Tuple[ExecutionTrace, Dict[str, Any]]: The validated trace and output dict.
    """
    from tensordict import TensorDict

    from logic.src.envs.cmptp import CMPTPEnv
    from logic.src.models.core.attention_model import AttentionModel

    torch.manual_seed(seed)
    if model is None:
        model = AttentionModel(embed_dim=32, hidden_dim=64, problem=problem, n_encode_layers=1, n_heads=4)
        model.eval()

    if env is None:
        env = CMPTPEnv(generator_params={"num_loc": graph_size}, check_env_specs=False)

    if td is None:
        g = torch.Generator().manual_seed(seed)
        td = TensorDict(
            {
                "locs": torch.rand(1, graph_size, 2, generator=g),
                "depot": torch.tensor([[0.5, 0.5]]),
                "waste": torch.rand(1, graph_size, generator=g) * 0.8 + 0.1,
                "capacity": torch.tensor([2.0]),
            },
            batch_size=[1],
        )
        td = env.reset(td)

    out = model(td, env, strategy=strategy, trace=True, seed=seed)
    return out["trace"], out
