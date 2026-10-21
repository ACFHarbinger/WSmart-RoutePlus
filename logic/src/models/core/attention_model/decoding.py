"""Decoding strategies for Attention Models.

This module provides common decoding logic used by attention-based constructive
policies, including greedy selection, sampling, and advanced search methods
like Beam Search and POMO.

Attributes:
    DecodingMixin: Mixin class providing decoding-related utilities.

Example:
    >>> mixin = DecodingMixin()
    >>> mixin.set_strategy("sampling", temp=0.8)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


class DecodingMixin:
    """Mixin for managing constructive decoding strategies.

    Provides high-level interfaces for controlling how the model selects actions
    during solution construction (e.g., greedy, sampling, or search). This mixin
    is intended to be inherited by models that use the common decoder subnets.

    Attributes:
        strategy (str): Current selection mode ('greedy', 'sampling', 'beam').
        temp (Optional[float]): Softmax temperature for sampling diversity.
    """

    def __init__(self) -> None:
        """Initializes the decoding mixin state."""
        self.strategy: str = "greedy"
        self.temp: Optional[float] = None

    def set_strategy(self, strategy: str, temp: Optional[float] = None) -> None:
        """Configures the current action selection strategy.

        Args:
            strategy: Identifier for the decoding tactic ('greedy' or 'sampling').
            temp: Scaling factor for sampling probabilities. Higher values
                increase exploration.
        """
        self.strategy = strategy
        if temp is not None:
            self.temp = temp

    def beam_search(self, *args: Any, **kwargs: Any) -> Any:
        """Executes a beam search exploration.

        Delegates the search logic to the problem instance, providing the
        model itself as the transition heuristic.

        Args:
            args: Positional arguments for the problem-specific beam search.
            kwargs: Keyword arguments for beam width, depth, etc.

        Returns:
            Any: The best tour discovered within the beam width.
        """
        return self.problem.beam_search(*args, **kwargs, model=self)

    def propose_expansions(
        self,
        beam: Any,
        fixed: Any,
        expand_size: Optional[int] = None,
        normalize: bool = False,
        max_calc_batch_size: int = 4096,
    ) -> Any:
        """Generates candidate nodes for beam expansion.

        This method bridges the beam search logic and the model's decoder,
        typically requesting log-probabilities for the next step.

        Args:
            beam: Current state of the construction beam.
            fixed: Pre-encoded graph embeddings.
            expand_size: Number of top-k candidates to propose.
            normalize: Whether to re-normalize probabilities after masking.
            max_calc_batch_size: Throughput cap for parallel expansion.

        Returns:
            Any: A structure containing predicted indices and their log-probs.
        """
        # Cache rows belong to original instances, not to current beam positions.
        # Index even when batch sizes match: pruning can reorder/duplicate rows.
        fixed = fixed[beam.state.ids.reshape(-1)]
        if hasattr(self.decoder, "propose_expansions"):
            log_p = self.decoder.propose_expansions(beam, fixed, expand_size, normalize, max_calc_batch_size)
        else:
            log_p, _ = self.decoder._get_log_p(fixed, beam.state, normalize=normalize)
        return log_p.squeeze(1) if log_p.dim() == 3 and log_p.size(1) == 1 else log_p

    def sample_many(
        self,
        input: Dict[str, Any],
        cost_weights: Optional[torch.Tensor] = None,
        batch_rep: int = 1,
        iter_rep: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Constructs multiple solutions for the same input instances.

        Useful for validation, bootstrapping, or the POMO (Parallelized Open
        Multi-Start) algorithm.

        Args:
            input: Dictionry/TensorDict of problem data.
            cost_weights: Scaling factors for objectives in multi-objective cases.
            batch_rep: Number of parallel construction starts per batch element.
            iter_rep: Number of sequential sampling passes per batch element.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - costs (torch.Tensor): Calculated path lengths [batch, ...].
                - pis (torch.Tensor): Tour sequences [batch, nodes].
        """
        from logic.src.utils.functions import sample_many as _sample_many

        return _sample_many(
            lambda i: self.forward(
                i,
                strategy="sampling",
                return_pi=True,
                cost_weights=cost_weights,
            ),
            input,  # type: ignore[arg-type]
            batch_rep,
            iter_rep,
        )
