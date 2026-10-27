"""Constructive decoding loop for GlimpseDecoder.

Moved from ``logic.src.models.subnets.decoders.glimpse.decoder``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch

from logic.src.models.subnets.decoders.common import AttentionDecoderCache


class GlimpseDecoderInnerMixin:
    """Moved methods of :class:`GlimpseDecoder`."""

    problem: Any
    strategy: Optional[str]
    generator: torch.Generator
    seed: int

    def _precompute(self, embeddings: torch.Tensor, num_steps: int = 1) -> AttentionDecoderCache:
        """Provided by :class:`GlimpseDecoderLogitsMixin`."""
        raise NotImplementedError

    def _get_log_p(
        self,
        fixed: AttentionDecoderCache,
        state: Any,
        normalize: bool = True,
        mask_val: float = float("-inf"),
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Provided by :class:`GlimpseDecoderLogitsMixin`."""
        raise NotImplementedError

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
        if hasattr(self.problem, "evaluate_tour"):
            out_eval = self.problem.evaluate_tour(nodes, pi)
            if isinstance(out_eval, dict):
                cost = out_eval.get("cost")
                final_td = out_eval.get("td")
            elif isinstance(out_eval, tuple):
                cost = out_eval[0]
                if len(out_eval) > 2:
                    final_td = out_eval[2]
            else:
                cost = out_eval
        elif hasattr(self.problem, "get_costs"):
            out_cost = self.problem.get_costs(nodes, pi, None)
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
