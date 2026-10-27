"""Decoder based on Multi-Head Attention (MHA).

Logit helpers live in ``decoder_logits``.

The inner decoding loop lives in ``decoder_inner``.

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

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from logic.src.models.subnets.embeddings import CONTEXT_EMBEDDING_REGISTRY

from .decoder_inner import (
    GlimpseDecoderInnerMixin as GlimpseDecoderInnerMixin,
)
from .decoder_logits import (
    GlimpseDecoderLogitsMixin as GlimpseDecoderLogitsMixin,
)


class GlimpseDecoder(GlimpseDecoderLogitsMixin, GlimpseDecoderInnerMixin, nn.Module):
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
            self.context_embedding: Optional[nn.Module] = CONTEXT_EMBEDDING_REGISTRY[env_name](
                embed_dim=embed_dim, temporal_horizon=kwargs.get("temporal_horizon", 0)
            )
            self.project_step_context: nn.Module = nn.Identity()
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
