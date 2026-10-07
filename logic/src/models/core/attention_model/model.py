"""Attention Model (AM) core module.

This module provides the implementation of the Attention Model (Kool et al. 2019),
a graph-based neural network that uses multi-head attention to constructively
solve Vehicle Routing Problems. It supports various problem domains
including TSP, VRPP, CVRPP and CTOP.

Attributes:
    AttentionModel: The primary constructive neural routing policy.

Example:
    >>> from logic.src.models.core.attention_model.model import AttentionModel
    >>> model = AttentionModel(embed_dim=128, hidden_dim=512, problem="vrpp")
    >>> out = model(td, env, strategy="greedy")
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.utils.checkpoint
from tensordict import TensorDict
from torch import nn

from logic.src.configs.models.activation_function import ActivationConfig
from logic.src.configs.models.normalization import NormalizationConfig
from logic.src.constants.models import (
    FEED_FORWARD_EXPANSION,
    TANH_CLIPPING,
)
from logic.src.envs.base.base import RL4COEnvBase
from logic.src.interfaces.tensor_dict_like import ITensorDictLike
from logic.src.models.core.attention_model.decoding import DecodingMixin
from logic.src.models.core.attention_model.policy import AttentionModelPolicy
from logic.src.models.subnets.embeddings import get_init_embedding
from logic.src.models.subnets.factories import NeuralComponentFactory
from logic.src.utils.functions.problem import is_tsp_problem, is_vrpp_problem


class _ContextEmbedderAdapter:
    """Thin proxy delegating context embedding calls to an underlying module.

    Preserves backwards compatibility for callers expecting the legacy
    ``context_embedder`` interface without registering duplicate parameter
    keys in the model's ``state_dict``.
    """

    def __init__(self, target: nn.Module) -> None:
        self.__dict__["_target"] = target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._target(*args, **kwargs)

    def init_node_embeddings(self, nodes: Any, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self._target, "init_node_embeddings"):
            return self._target.init_node_embeddings(nodes, *args, **kwargs)
        return self.forward(nodes, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # deepcopy probes a newly allocated proxy before restoring its state.
        # Bypass __getattr__ while looking up the target so that this probe
        # raises AttributeError instead of recursively querying _target.
        target = object.__getattribute__(self, "_target")
        return getattr(target, name)

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        # Do not delegate copy/state protocols to the wrapped nn.Module.
        # The shared memo also preserves the alias to a copied model's
        # registered init_embedding rather than creating a second module.
        cloned = object.__new__(type(self))
        memo[id(self)] = cloned
        cloned.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return cloned

    def __setattr__(self, name: str, val: Any) -> None:
        if name == "_target":
            self.__dict__["_target"] = val
        else:
            self.__dict__[name] = val


class AttentionModel(AttentionModelPolicy, DecodingMixin):
    """Attention Model for neural combinatorial optimization.

    Unified constructive routing policy combining Graph Attention Network (GAT)
    encoding and autoregressive glimpse-based decoding. Subclasses
    ``AttentionModelPolicy`` to ensure full weight-level and algorithmic parity
    between training and evaluation while preserving backwards compatibility with
    legacy checkpoint layouts and decoding APIs.

    Attributes:
        embed_dim (int): Dimensionality of node and graph embeddings.
        hidden_dim (int): Hidden dimension for feed-forward sublayers.
        n_heads (int): Number of multi-head attention heads.
        problem (Any): Problem definition or environment object.
        encoder (nn.Module): The graph attention encoder.
        decoder (nn.Module): The autoregressive attention decoder.
        init_embedding (nn.Module): Problem-specific initial feature projection.
        context_embedder (Any): Backwards-compatible proxy to ``init_embedding``.
        pomo_size (int): Parallel start size for POMO (if enabled).
        checkpoint_encoder (bool): Whether to use gradient checkpointing on encoding.
        aggregation_graph (str): Global aggregation method ('avg', 'max', 'sum').
        temporal_horizon (int): Number of future steps to consider for dynamics.
        tanh_clipping (float): Logit clipping value for stable training.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        problem: Any = "vrpp",
        component_factory: Optional[NeuralComponentFactory] = None,
        n_encode_layers: int = 3,
        n_encode_sublayers: Optional[int] = None,
        n_decode_layers: Optional[int] = None,
        dropout_rate: float = 0.1,
        aggregation: str = "sum",
        aggregation_graph: str = "avg",
        tanh_clipping: float = TANH_CLIPPING,
        mask_inner: bool = True,
        mask_logits: bool = True,
        mask_graph: bool = False,
        norm_config: Optional[NormalizationConfig] = None,
        activation_config: Optional[ActivationConfig] = None,
        n_heads: int = 8,
        checkpoint_encoder: bool = False,
        shrink_size: Optional[int] = None,
        pomo_size: int = 0,
        temporal_horizon: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        entropy_weight: float = 0.0,
        predictor_layers: Optional[int] = None,
        connection_type: str = "residual",
        hyper_expansion: int = FEED_FORWARD_EXPANSION,
        decoder_type: str = "attention",
        **kwargs: Any,
    ) -> None:
        """Initializes the unified Attention Model."""
        if isinstance(problem, str):
            env_name = problem.lower()
        elif hasattr(problem, "NAME") and isinstance(problem.NAME, str):
            env_name = problem.NAME.lower()
        elif hasattr(problem, "name") and isinstance(problem.name, str):
            env_name = problem.name.lower()
        elif is_vrpp_problem(problem):
            env_name = "vrpp"
        elif is_tsp_problem(problem):
            env_name = "tsp"
        else:
            env_name = "vrpp"

        if norm_config is None:
            norm_config = NormalizationConfig(
                norm_type=kwargs.get("normalization", "batch"),
                learn_affine=kwargs.get("norm_learn_affine", kwargs.get("learn_affine", True)),
                track_stats=kwargs.get("norm_track_stats", kwargs.get("track_stats", False)),
                epsilon=kwargs.get("norm_eps_alpha", kwargs.get("epsilon_alpha", 1e-5)),
                momentum=kwargs.get("norm_momentum_beta", kwargs.get("momentum_beta", 0.1)),
                k_lrnorm=kwargs.get("lrnorm_k", 1),
                n_groups=kwargs.get("gnorm_groups", 1),
            )

        if activation_config is None:
            activation_config = self._resolve_activation_config(component_factory is not None, kwargs)

        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_encode_layers=n_encode_layers,
            n_heads=n_heads,
            norm_config=norm_config,
            activation_config=activation_config,
            **kwargs,
        )
        DecodingMixin.__init__(self)

        # If custom factory provided, use it to instantiate encoder and decoder
        if component_factory is not None:
            enc = component_factory.create_encoder(
                embed_dim=embed_dim,
                feed_forward_hidden=hidden_dim,
                n_layers=n_encode_layers,
                n_sublayers=n_encode_sublayers,
                norm_config=norm_config,
                activation_config=activation_config,
                dropout_rate=dropout_rate,
                aggregation=aggregation,
                hyper_expansion=hyper_expansion,
                connection_type=connection_type,
                n_heads=n_heads,
                mask_inner=mask_inner,
                mask_graph=mask_graph,
                spatial_bias=spatial_bias,
                spatial_bias_scale=spatial_bias_scale,
            )
            step_context_dim = 2 * embed_dim + embed_dim
            dec = component_factory.create_decoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                problem=problem,
                n_layers=n_decode_layers,
                n_heads=n_heads,
                step_context_dim=step_context_dim,
                predictor_layers=predictor_layers,
                norm_config=norm_config,
                activation_config=activation_config,
                dropout_rate=dropout_rate,
                aggregation=aggregation,
                hyper_expansion=hyper_expansion,
                connection_type=connection_type,
                tanh_clipping=tanh_clipping,
                mask_logits=mask_logits,
                shrink_size=shrink_size,
                decoder_type=decoder_type,
            )
            if isinstance(enc, nn.Module) or enc is None:
                self.encoder = enc
            else:
                if "encoder" in self._modules:
                    del self._modules["encoder"]
                self.__dict__["encoder"] = enc

            if isinstance(dec, nn.Module) or dec is None:
                self.decoder = dec
            else:
                if "decoder" in self._modules:
                    del self._modules["decoder"]
                self.__dict__["decoder"] = dec

        self.problem = problem
        self.pomo_size = pomo_size
        self.checkpoint_encoder = checkpoint_encoder
        self.aggregation_graph = aggregation_graph
        self.temporal_horizon = temporal_horizon
        self.tanh_clipping = tanh_clipping
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self._configure_legacy_embedding(temporal_horizon, embed_dim)
        self._context_embedder_adapter: Optional[_ContextEmbedderAdapter] = None
        self._context_embedder_override: Optional[Any] = None
        self.set_strategy("greedy")

    @staticmethod
    def _resolve_activation_config(legacy_factory: bool, kwargs: Dict[str, Any]) -> ActivationConfig:
        """Retain legacy factory defaults while honoring explicit activation overrides."""
        defaults = (
            ActivationConfig()
            if legacy_factory
            else ActivationConfig(name="relu", param=0.0, threshold=0.0, replacement_value=0.0, n_params=1)
        )
        range_val = kwargs.get("af_uniform_range", kwargs.get("af_urange"))
        return ActivationConfig(
            name=kwargs.get("activation_function", kwargs.get("activation", defaults.name)),
            param=kwargs.get("af_param", defaults.param),
            threshold=kwargs.get("af_threshold", defaults.threshold),
            replacement_value=kwargs.get(
                "af_replacement_value", kwargs.get("af_replacement", defaults.replacement_value)
            ),
            n_params=kwargs.get("af_num_params", kwargs.get("af_nparams", defaults.n_params)),
            range=list(range_val) if isinstance(range_val, (list, tuple)) else defaults.range,
        )

    def _configure_legacy_embedding(self, temporal_horizon: int, embed_dim: int) -> None:
        """Preserve legacy feature width and depot projection without changing policy defaults."""
        if temporal_horizon > 0 and self.env_name is not None:
            self.init_embedding = get_init_embedding(
                self.env_name,
                embed_dim,
                temporal_horizon=temporal_horizon,
            )
        # Legacy checkpoints projected concatenated depots with the node layer.
        # Keep this compatibility behavior local to AttentionModel.
        if hasattr(self.init_embedding, "legacy_depot_projection"):
            self.init_embedding.legacy_depot_projection = True

    @property
    def is_vrpp(self) -> bool:
        """Determines if the model is configured for VRP with Profits."""
        return is_vrpp_problem(self.problem)

    @property
    def context_embedder(self) -> Any:
        """Backwards-compatible proxy to the initial embedding module."""
        if self._context_embedder_override is not None:
            return self._context_embedder_override
        if self._context_embedder_adapter is None:
            self._context_embedder_adapter = _ContextEmbedderAdapter(self.init_embedding)
        return self._context_embedder_adapter

    @context_embedder.setter
    def context_embedder(self, val: Any) -> None:
        self._context_embedder_override = val

    def _get_initial_embeddings(self, input: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Processes raw instance data into initial node embeddings."""
        res = self.context_embedder(input)
        if isinstance(res, tuple):
            return res
        return res, None

    def _aggregate_graph_context(self, outputs: torch.Tensor) -> torch.Tensor:
        """Aggregates per-node embeddings into a summary graph-level representation."""
        if self.aggregation_graph in ("avg", "mean"):
            return outputs.mean(dim=1)
        if self.aggregation_graph == "max":
            return outputs.max(dim=1)[0]
        if self.aggregation_graph == "sum":
            return outputs.sum(dim=1)
        return outputs.mean(dim=1)

    def precompute_fixed(self, input: Any, edges: Optional[torch.Tensor] = None) -> Any:
        """Precomputes and caches graph-level state for efficient search."""
        from logic.src.utils.decoding import CachedLookup

        embeddings, init_context = self._get_initial_embeddings(input)
        encoded_embeddings = self.encoder(embeddings)
        self.decoder(input, encoded_embeddings, init_context, None, precompute_only=True)
        return CachedLookup(embeddings=encoded_embeddings, context=init_context)

    def expand(self, t: Union[torch.Tensor, ITensorDictLike, None]) -> Any:
        """Expands instance features to support parallel POMO constructions."""
        if t is None:
            return None
        if isinstance(t, ITensorDictLike):
            return t.__class__({k: self.expand(v) for k, v in t.items()})  # type: ignore[call-arg]

        bs = t.size(0)
        shape = (bs, self.pomo_size) + t.shape[1:]
        return t.unsqueeze(1).expand(shape).reshape(-1, *t.shape[1:])

    def forward(  # type: ignore[override]
        self,
        td: Union[TensorDict, Dict[str, Any]],
        env: Optional[Any] = None,
        strategy: Optional[str] = None,
        num_starts: int = 1,
        actions: Optional[torch.Tensor] = None,
        start_nodes: Optional[torch.Tensor] = None,
        return_pi: bool = False,
        pad: bool = False,
        mask: Optional[torch.Tensor] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes constructive routing search.

        Supports both RL4CO environments (standard policy execution) and legacy
        constructive decoder workflows.
        """
        strat = strategy or self.strategy

        # 1. RL4CO environment execution
        if env is not None and isinstance(env, RL4COEnvBase):
            if not isinstance(td, TensorDict):
                batch_size = td.get("locs", td.get("loc", td.get("depot"))).size(0)
                td_tensor = TensorDict(td, batch_size=[batch_size])
            else:
                td_tensor = td
            out = super().forward(
                td=td_tensor,
                env=env,
                strategy=strat,
                num_starts=num_starts,
                actions=actions,
                start_nodes=start_nodes,
                **kwargs,
            )
            if "cost" not in out and "reward" in out:
                out["cost"] = -out["reward"]
            if "pi" not in out and "actions" in out:
                out["pi"] = out["actions"]
            if "log_p" not in out and "log_likelihood" in out:
                out["log_p"] = out["log_likelihood"]
            return out

        # 2. Legacy / Mock constructive execution
        embeddings, init_context = self._get_initial_embeddings(td)
        if self.checkpoint_encoder and self.training:
            outputs = cast(
                torch.Tensor,
                torch.utils.checkpoint.checkpoint(self.encoder, embeddings, mask, use_reentrant=False),
            )
        else:
            outputs = self.encoder(embeddings, mask)

        graph_context = self._aggregate_graph_context(outputs)
        out_dec = self.decoder(
            td,
            outputs,
            graph_context,
            init_context,
            env,
            strategy=strat,
            return_pi=return_pi,
            expert_pi=expert_pi,
        )

        if isinstance(out_dec, tuple):
            if len(out_dec) == 4:
                _log_p, pi, cost, final_td = out_dec
            elif len(out_dec) == 3:
                _log_p, pi, cost = out_dec
                final_td = None
            else:
                _log_p = out_dec[0]
                pi = out_dec[1] if len(out_dec) > 1 else None
                cost = out_dec[2] if len(out_dec) > 2 else None
                final_td = out_dec[3] if len(out_dec) > 3 else None
        else:
            _log_p = out_dec
            pi = None
            cost = None
            final_td = None

        reward = (
            -cost
            if cost is not None
            else torch.tensor(0.0, device=outputs.device if isinstance(outputs, torch.Tensor) else None)
        )
        out = {"cost": cost, "reward": reward, "td": final_td}
        if _log_p is not None:
            out["log_likelihood"] = _log_p
            out["log_p"] = _log_p
        if pi is not None:
            out["actions"] = pi
            if return_pi:
                out["pi"] = pi
        return out

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Loads state dictionary with key remapping at any nesting level."""
        _REMAP = {
            f"{prefix}context_embedder.init_embed.": f"{prefix}init_embedding.node_embed.",
            f"{prefix}context_embedder.init_embed_depot.": f"{prefix}init_embedding.depot_embed.",
        }
        dead_prefixes = (
            f"{prefix}context_embedder.project_step_context.",
            f"{prefix}decoder.project_fixed_context.",
            f"{prefix}project_fixed_context.",
        )
        for old_pfx, new_pfx in _REMAP.items():
            matching_keys = [k for k in list(state_dict.keys()) if k.startswith(old_pfx)]
            for k in matching_keys:
                suffix = k[len(old_pfx) :]
                new_key = f"{new_pfx}{suffix}"
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(k)

        for dp in dead_prefixes:
            matching_keys = [k for k in list(state_dict.keys()) if k.startswith(dp)]
            for k in matching_keys:
                state_dict.pop(k)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        """Loads state dictionary with automatic key remapping for legacy checkpoints.

        Strips Lightning ``policy.`` prefix if present at root level, then delegates
        to PyTorch's loading mechanism (which invokes ``_load_from_state_dict``).
        """
        if any(k.startswith("policy.") for k in state_dict.keys()):
            state_dict = {(k[len("policy.") :] if k.startswith("policy.") else k): v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)
