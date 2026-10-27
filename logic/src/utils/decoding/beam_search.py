"""
Beam search module.

Attributes:
    BeamSearch: Beam search decoding strategy.
    beam_search: Beam search decoding strategy.
    get_beam_search_results: Get the results of beam search.
    _beam_search: Internal beam search execution.

Example:
    >>> from logic.src.utils.decoding import BeamSearch, beam_search, get_beam_search_results, _beam_search
    >>> beam_search = BeamSearch()
    >>> beam_search = beam_search(state, beam_size)
    >>> beam_search = get_beam_search_results(beams, final_state)
    >>> beam_search = _beam_search(state, beam_size)
"""

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from .base import DecodingStrategy
from .batch_beam import BatchBeam
from .decoding_utils import backtrack


class BeamSearch(DecodingStrategy):
    """
    Beam search decoding: maintain top-k partial solutions.

    Note: Beam search requires special handling in the policy forward pass.
    This implementation provides the step function for scoring candidates.

    Attributes:
        beam_width (int): Number of beams to maintain.
    """

    def __init__(self, beam_width: int = 5, **kwargs) -> None:
        """
        Initialize BeamSearch decoding.

        Args:
            beam_width: Number of beams to maintain.
            kwargs: Passed to super class.
        """
        super().__init__(**kwargs)
        self.beam_width = beam_width

    def step(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        td: Optional[TensorDict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k actions for beam search.

        Args:
            logits: Logits from the policy.
            mask: Mask for the logits.
            td: TensorDict containing the state.

        Returns:
            Tuple of (top_k_actions, top_k_log_probs, entropy) each with shape [batch, beam_width]
        """
        logits = self._process_logits(logits, mask)
        log_probs = F.log_softmax(logits, dim=-1)

        # Get top-k
        top_log_probs, top_actions = torch.topk(log_probs, self.beam_width, dim=-1)

        # Placeholder entropy for beam search (complex to define per step)
        entropy = torch.zeros_like(top_log_probs)

        return top_actions, top_log_probs, entropy


def beam_search(*args, **kwargs):
    """
    Orchestrates beam search decoding on a batch of states.
    This is the high-level entry point that runs the search and then reconstructs the results.

    Args:
        args: positional args passed to _beam_search
        kwargs: keyword args passed to _beam_search

    Returns:
        tuple: (score, solutions, cost, ids, batch_size)
    """
    # 1. Execute the search loop to get a list of beams and the final state
    beams, final_state = _beam_search(*args, **kwargs)

    # 2. Extract and format the results from the raw beam data
    return get_beam_search_results(beams, final_state)


def get_beam_search_results(beams: list[BatchBeam], final_state: Any) -> tuple:
    """
    Reconstructs solutions from the final beam state.
    This involves backtracking through the beam parents to reconstruct the full path.

    Args:
        beams: List of BatchBeam objects from each step.
        final_state: The final state after decoding.

    Returns:
        tuple: (score, solutions, cost, ids, batch_size)
    """
    beam = beams[-1]  # Final beam containing top-k results
    if final_state is None:
        return None, None, None, None, beam.batch_size

    # Backtracking Phase:
    # 1. Gather all actions and parents taken at each step.
    # First state has no actions/parents and should be omitted when backtracking.
    actions = [beam.action for beam in beams[1:] if beam.action is not None]
    parents = [beam.parent for beam in beams[1:] if beam.parent is not None]

    # 2. Use the backtrack utility to resolve the sequence for each candidate in the top-k.
    # 3. Use the environment state to convert these action sequences into a problem-specific solution format.
    solutions = final_state.construct_solutions(backtrack(parents, actions))

    return (
        beam.score,
        solutions,
        final_state.get_final_cost()[:, 0],
        final_state.ids.view(-1),
        beam.batch_size,
    )


def _beam_search(
    state: Any, beam_size: int, propose_expansions: Optional[Callable] = None, keep_states: bool = False
) -> tuple:
    """
    Internal beam search execution.

    Args:
        state: Initial state.
        beam_size: Width of the beam.
        propose_expansions: Custom expansion logic.
        keep_states: Whether to store all intermediate states.

    Returns:
        tuple: (beams, final_state)
    """
    # 1. Initialize the beam with the starting state
    # BatchBeam manages a batch of beams, where each beam has 'beam_size' candidates.
    beam = BatchBeam.initialize(state)

    # Initial state tracking
    beams = [beam if keep_states else beam.clear_state()]

    # 2. Iterative decoding until all beams reach the terminal state (e.g., all nodes visited)
    while not beam.all_finished():
        # Expansion Phase:
        # The model (or a heuristic) proposes the best next actions for each candidate in each beam.
        # This typically returns a set of (parent_index, action, score) triples.
        parent, action, score = beam.propose_expansions() if propose_expansions is None else propose_expansions(beam)

        if parent is None:
            # Handle cases where no valid expansions are possible
            return beams, None

        # 3. Expansion:
        # Apply the chosen actions to create new states. This expands the beam width
        # from B to (B * K) where K is the number of proposed expansions.
        beam = beam.expand(parent, action, score=score)

        # 4. Pruning (Selection Phase):
        # Keep only the top 'beam_size' best candidates for each element in the batch.
        # This prevents the exponential growth of the search space.
        beam = beam.topk(beam_size)

        # 5. Collection:
        # Store intermediate results. If keep_states=False, we discard the State objects
        # to save memory, keeping only the parent/action pointers for backtracking.
        beams.append(beam if keep_states else beam.clear_state())

    # Return the final state separately since beams may not keep state
    return beams, beam.state
