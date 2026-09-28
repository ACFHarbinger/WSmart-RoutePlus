"""Decoder based on Multi-Head Attention (MHA).

This module implements a constructive decoder that uses multi-head attention
with a glimpse mechanism to autonomously select nodes in routing problems.

Attributes:
    GlimpseDecoder: Autoregressive decoder using MHA and glimpse mechanisms.

Example:
    >>> from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder
    >>> decoder = GlimpseDecoder(embed_dim=128, hidden_dim=512, problem=tsp_env)
    >>> log_p, pi, cost, _ = decoder(input_tensor, node_embeddings)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from logic.src.models.subnets.decoders.common import AttentionDecoderCache
from logic.src.models.subnets.decoders.glimpse.attention import make_heads, one_to_many_logits
from logic.src.models.subnets.embeddings import CONTEXT_EMBEDDING_REGISTRY


class GlimpseDecoder(nn.Module):
    """Decoder based on Multi-Head Attention (MHA).

    Attributes:
        embed_dim (int): Dimensionality of embeddings.
        problem (Any): Problem environment for state management.
        n_heads (int): Number of attention heads.
        mask_inner (bool): Whether to mask visited nodes in glimpse.
        mask_logits (bool): Whether to mask visited nodes in final probabilities.
        tanh_clipping (float): Clipping range for logits.
        mask_graph (bool): Whether to apply graph-level masking.
        shrink_size (Optional[int]): Optimization parameter for state shrinking.
        pomo_size (int): Size for POMO (Parallel Optimal Model Optimization).
        spatial_bias (bool): Whether to apply spatial bias in attention.
        spatial_bias_scale (float): Scaling factor for spatial bias.
        strategy (Optional[str]): Decoding strategy (e.g., "greedy", "sampling").
        seed (int): Random seed for reproducibility.
        generator (torch.Generator): Random number generator for selection.
        project_node_embeddings (nn.Linear): Linear projection for V/K/Q.
        project_step_context (nn.Linear): Projection for the dynamic step context.
        step_context_dim (int): Dimensionality of the step context embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        problem: Any,
        n_heads: int = 8,
        mask_inner: bool = True,
        mask_logits: bool = True,
        tanh_clipping: float = 10.0,
        mask_graph: bool = False,
        shrink_size: Optional[int] = None,
        pomo_size: int = 0,
        spatial_bias: bool = False,
        spatial_bias_scale: float = 1.0,
        strategy: Optional[str] = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initializes class.

        Args:
            embed_dim: Node embedding dimensionality.
            hidden_dim: Hidden dimension size for layers.
            problem: Problem environment class.
            n_heads: Number of attention heads.
            mask_inner: Whether to mask inner attention.
            mask_logits: Whether to mask output logits.
            tanh_clipping: Logit clipping value.
            mask_graph: Whether to use graph masking.
            shrink_size: Batch shrinkage parameter.
            pomo_size: Number of POMO augmentation starts.
            spatial_bias: Apply spatial bias flag.
            spatial_bias_scale: Scale for spatial bias.
            strategy: Default decoding strategy.
            seed: Initialization seed for selection.
            kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.problem = problem
        self.n_heads = n_heads
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.tanh_clipping = tanh_clipping
        self.mask_graph = mask_graph
        self.shrink_size = shrink_size
        self.pomo_size = pomo_size
        self.spatial_bias = spatial_bias
        self.spatial_bias_scale = spatial_bias_scale
        self.strategy = strategy
        self.seed = seed
        device = kwargs.get("device", "cpu")
        self.generator = torch.Generator(device=device).manual_seed(self.seed)

        # Project node embeddings to MHA heads
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        if isinstance(self.problem, str):
            env_name = self.problem.lower()
        else:
            env_name = getattr(self.problem, "name", getattr(self.problem, "NAME", "tsp")).lower()

        if env_name in CONTEXT_EMBEDDING_REGISTRY:
            self.context_embedding = CONTEXT_EMBEDDING_REGISTRY[env_name](
                embed_dim=embed_dim, temporal_horizon=kwargs.get("temporal_horizon", 0)
            )
            self.project_step_context = nn.Identity()
            self.step_context_dim = self.context_embedding.step_context_dim
        else:
            self.context_embedding = None
            self.project_step_context = nn.Linear(embed_dim, embed_dim, bias=False)
            self.step_context_dim = embed_dim

    @property
    def device(self) -> torch.device:
        """Gets the device of the model.

        Returns:
            torch.device: Model device.
        """
        return next(self.parameters()).device

    def set_step_context_dim(self, dim: int) -> None:
        """Sets the dimension of the step context projection.

        Args:
            dim: New dimensionality for step context.
        """
        self.step_context_dim = dim
        self.project_step_context = nn.Linear(dim, self.embed_dim, bias=False)

    def set_strategy(self, strategy: str, temp: Optional[float] = None) -> None:
        """Sets the decoding strategy and temperature.

        Args:
            strategy: Strategy name (e.g., "greedy").
            temp: Softmax temperature (optional).
        """
        self.strategy = strategy
        if temp is not None:
            self.temp = temp

    def forward(
        self,
        input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        embeddings: torch.Tensor,
        fixed_context: Optional[torch.Tensor] = None,
        init_context: Optional[torch.Tensor] = None,
        env: Optional[Any] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """Standard Module forward wrapper.

        Args:
            input: Input features or state dictionary.
            embeddings: Contextual node embeddings.
            fixed_context: Precomputed fixed context.
            init_context: Initial context.
            env: Environment instance.
            expert_pi: Expert actions.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple: Log probabilities, actions, costs, and final internal state.
        """
        return self._inner(
            input,
            embeddings,
            fixed_context,
            init_context,
            env,
            expert_pi,
            **kwargs,
        )

    def _inner(
        self,
        nodes: Union[torch.Tensor, Dict[str, torch.Tensor]],
        embeddings: torch.Tensor,
        fixed_context: Optional[torch.Tensor] = None,
        init_context: Optional[torch.Tensor] = None,
        env: Optional[Any] = None,
        expert_pi: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """Constructive decoding loop.

        Args:
            nodes: Input features of the nodes.
            embeddings: Contextual embeddings from the encoder.
            fixed_context: Fixed context for the problem.
            init_context: Initial step context.
            env: Environment class instance.
            expert_pi: Ground truth expert actions.
            kwargs: Additional keywords.

        Returns:
            Tuple: Log probabilities, action sequences, costs, and final state.
        """
        outputs = []
        sequences = []

        # cost_weights and dist_matrix can be passed via kwargs or init_context
        cost_weights = kwargs.get("cost_weights")
        dist_matrix = kwargs.get("dist_matrix")

        state = self.problem.make_state(nodes, None, cost_weights, dist_matrix, **kwargs)
        fixed = self._precompute(embeddings)

        # Allow overriding strategy via kwargs (e.g. from AttentionModel.forward)
        strategy_name = kwargs.get("strategy", self.strategy) or "greedy"

        # Mask can be passed via kwargs
        mask = kwargs.get("mask")

        # Try to get graph size for safety break
        try:
            if isinstance(nodes, torch.Tensor):
                graph_size = nodes.shape[1]
            elif hasattr(nodes, "get"):
                # Handle TensorDict or dict
                loc_tensor = nodes.get("locs") if "locs" in nodes.keys() else nodes.get("loc", None)
                graph_size = loc_tensor.shape[1] if hasattr(loc_tensor, "shape") else 100  # type: ignore[union-attr]
            else:
                graph_size = 100
        except Exception:
            graph_size = 100

        # Safety break for infinite loops (e.g. 10x graph size)
        max_steps = max(100, graph_size * 10)

        i = 0
        while not state.all_finished() and i < max_steps:
            log_p, current_mask = self._get_log_p(fixed, state, mask=mask)
            selected = self._select_node(log_p.exp(), current_mask, strategy=strategy_name)

            state = state.update(selected)

            outputs.append(log_p)
            sequences.append(selected)
            i += 1

        if i >= max_steps:
            print(f" [!] Warning: Decoding reached max_steps ({max_steps}). Possible infinite loop.")

        # Stack outputs
        _log_p = torch.stack(outputs, 1)
        pi = torch.stack(sequences, 1)

        # Calculate cost
        cost = None
        final_td = None
        if hasattr(self.problem, "get_costs"):
            out_cost = self.problem.get_costs(nodes, pi, None)
            # Handle tuple return (cost, mask, td)
            if isinstance(out_cost, tuple):
                cost = out_cost[0]
                if len(out_cost) > 2:
                    final_td = out_cost[2]
            else:
                cost = out_cost

        return _log_p, pi, cost, final_td

    def _select_node(self, probs: torch.Tensor, mask: Optional[torch.Tensor], strategy: str = "greedy") -> torch.Tensor:
        """Internal node selection logic.

        Args:
            probs: Probabilities for selecting the next node.
            mask: Valid node mask.
            strategy: Selection strategy name.

        Returns:
            torch.Tensor: Selected node indices.

        Raises:
            ValueError: If an unknown strategy is specified.
        """
        assert (probs == probs).all(), "Probs contain NaN"

        if strategy == "greedy":
            _, selected = probs.max(1)
            if mask is not None:
                # Handle potential 3D mask [B, 1, N] -> [B, N]
                curr_mask = mask
                if curr_mask.dim() == 3:
                    curr_mask = curr_mask.squeeze(1)
                assert not curr_mask.gather(1, selected.unsqueeze(-1)).any(), "Selected masked node"
        elif strategy == "sampling":
            # The generator is created at construction time; follow the model when it
            # is moved to another device (torch.multinomial requires matching devices).
            if self.generator.device.type != probs.device.type:
                self.generator = torch.Generator(device=probs.device).manual_seed(self.seed)

            curr_mask = mask  # type: ignore[assignment]
            if curr_mask is not None and curr_mask.dim() == 3:
                curr_mask = curr_mask.squeeze(1)

            if curr_mask is not None:
                # Draw once from the valid actions only. Rows whose valid actions carry no
                # probability mass fall back to a uniform draw over the valid actions.
                valid = ~curr_mask.bool()
                probs = probs.masked_fill(~valid, 0.0)
                row_mass = probs.sum(-1, keepdim=True)
                probs = torch.where(row_mass > 0, probs, valid.to(probs.dtype))
            selected = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)
        else:
            raise ValueError(f"Unknown decoding strategy: {strategy}")

        return selected

    def __getstate__(self) -> Dict[str, Any]:
        """Prepare state for pickling (handle non-picklable Generator).

        Returns:
            Dict[str, Any]: State dictionary with generator metadata.
        """
        state = self.__dict__.copy()
        # Generator is not picklable, save state as tensor
        state["generator_state"] = self.generator.get_state()
        state["generator_device"] = str(self.generator.device)
        del state["generator"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state after unpickling.

        Args:
            state: State dictionary containing generator state.
        """
        gen_state = state.pop("generator_state")
        gen_device = state.pop("generator_device")
        self.__dict__.update(state)
        # Restore generator
        self.generator = torch.Generator(device=gen_device)
        self.generator.set_state(gen_state)

    def _precompute(self, embeddings: torch.Tensor, num_steps: int = 1) -> AttentionDecoderCache:
        """Precompute K, V for the attention mechanism.

        Args:
            embeddings: Node embeddings from the encoder.
            num_steps: Unused.

        Returns:
            AttentionDecoderCache: Cache object containing keys and values.
        """
        node_embeddings = embeddings

        # Joint projection
        qkv = self.project_node_embeddings(node_embeddings)
        glimpse_K, glimpse_V, logit_K = qkv.chunk(3, dim=-1)

        fixed = AttentionDecoderCache(
            node_embeddings=node_embeddings,
            graph_context=None,
            glimpse_key=make_heads(glimpse_K, self.n_heads),
            glimpse_val=make_heads(glimpse_V, self.n_heads),
            logit_key=make_heads(logit_K, self.n_heads),
        )
        return fixed

    def _get_log_p(
        self,
        fixed: AttentionDecoderCache,
        state: Any,
        normalize: bool = True,
        mask_val: float = -math.inf,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for the current state.

        Args:
            fixed: Fixed keys/values in the attention mechanism.
            state: Current environment state.
            normalize: Whether to apply log-softmax.
            mask_val: Value for masked-out logits.
            mask: Optional additional mask.

        Returns:
            Tuple: Log probabilities and the resolved mask.

        Raises:
            ValueError: If no mask is available from state or argument.
        """
        query = self._get_parallel_step_context(fixed.node_embeddings, state)

        # 1. Resolve the mask and ensure it's not None for the type checker
        # We combine the state mask (visited nodes) with the optional external mask.
        current_mask = state.get_mask()
        if mask is not None:
            current_mask = current_mask | mask if current_mask is not None else mask

        if current_mask is None:
            raise ValueError("A mask must be provided either as an argument or via the state.")

        # 2. Handle the unsqueeze logic on the guaranteed tensor
        # We use a temporary variable so the analyzer sees 'Tensor', not 'Optional[Tensor]'
        input_mask = current_mask.unsqueeze(1) if current_mask.dim() == 2 else current_mask

        logits = one_to_many_logits(
            query.unsqueeze(1),
            fixed.glimpse_key,  # type: ignore[arg-type]
            fixed.glimpse_val,  # type: ignore[arg-type]
            fixed.logit_key,  # type: ignore[arg-type]
            input_mask,  # Now guaranteed to be a Tensor
            self.n_heads,
            tanh_clipping=self.tanh_clipping,
            mask_val=mask_val,
        ).squeeze(1)

        if normalize:
            return torch.log_softmax(logits, dim=-1), current_mask
        return logits, current_mask

    def _get_parallel_step_context(
        self, embeddings: torch.Tensor, state: Any, from_depot: bool = False
    ) -> torch.Tensor:
        """Extract step context from state and project.

        Args:
            embeddings: Node embeddings.
            state: Current environment state.
            from_depot: Flag for starting from depot.

        Returns:
            torch.Tensor: Projected step context query.
        """
        if hasattr(self, "context_embedding") and self.context_embedding is not None:
            step_context = self.context_embedding(embeddings, state)
            query = self.project_step_context(step_context)
            if query.dim() == 3:
                query = query.squeeze(1)
            return query

        current_node = state.get_current_node()
        batch_size = embeddings.size(0)

        # Ensure current_node is (batch, 1) or (batch)
        if current_node.dim() > 1:
            current_node = current_node.reshape(batch_size, -1)
            # If it became (batch, 1), good. If (batch, N), typically N=1 here.
            current_node = current_node[:, 0]  # Make it (batch,) for cleaner unsqueezing below

        # Simple context: [batch, embed]
        # AM typically uses [embeddings(current_node), embeddings(first_node), capacity_left]
        # Here we follow a simplified version matching the original code
        step_context = embeddings.gather(
            1, current_node.unsqueeze(1).unsqueeze(-1).expand(batch_size, 1, self.embed_dim)
        ).squeeze(1)

        return self.project_step_context(step_context)

    def _calc_log_likelihood(
        self,
        _log_p: torch.Tensor,
        a: torch.Tensor,
        mask: Optional[torch.Tensor],
        return_entropy: bool = False,
        kl_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Utility for log-likelihood and entropy calculation.

        Args:
            _log_p: Log probability predictions.
            a: Actions taken.
            mask: Valid sample mask.
            return_entropy: Whether to return entropy of distribution.
            kl_loss: Unused flag.

        Returns:
            Union: Log-likelihood or (log-likelihood, entropy).
        """
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        if mask is not None:
            log_p[mask] = 0

        if return_entropy:
            entropy = -(_log_p.exp() * _log_p).sum(2)
            if mask is not None:
                entropy[mask] = 0
            return log_p, entropy

        return log_p

    def propose_expansions(
        self,
        beam: Any,
        fixed: AttentionDecoderCache,
        expand_size: Optional[int] = None,
        normalize: bool = False,
        max_calc_batch_size: int = 4096,
    ) -> torch.Tensor:
        """Proposals for beam search.

        Args:
            beam: Beam object.
            fixed: Static attention cache.
            expand_size: Number of children to expand.
            normalize: Softmax normalization flag.
            max_calc_batch_size: Batching limit.

        Returns:
            torch.Tensor: Log probabilities for expansion candidates.
        """
        # Implementation depends on external Beam structure
        # Simplified for now
        log_p, _ = self._get_log_p(fixed, beam.state, normalize=normalize)
        return log_p
