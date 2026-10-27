"""
Shared utilities for decoding.

Attributes:
    top_k_filter: Filter logits to keep only top-k values.
    top_p_filter: Filter logits using nucleus (top-p) sampling.
    batchify: Repeat TensorDict for multistart.
    unbatchify: Unbatch a TensorDict/Tensor by extracting the original batch.
    gather_by_index: Gather elements from src using indices.
    get_log_likelihood: Get log likelihood of selected actions.
    modify_logits_for_top_k_filtering: Alias for top_k_filter for rl4co compatibility.
    modify_logits_for_top_p_filtering: Alias for top_p_filter for rl4co compatibility.

Example:
    >>> from logic.src.utils.decoding.decoding_utils import (
    ...     top_k_filter,
    ...     top_p_filter,
    ...     batchify,
    ...     unbatchify,
    ...     gather_by_index,
    ...     get_log_likelihood,
    ...     modify_logits_for_top_k_filtering,
    ...     modify_logits_for_top_p_filtering,
    ... )
"""

from __future__ import annotations

from typing import Optional, cast

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from logic.src.utils.functions.lexsort import torch_lexsort


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filter logits to keep only top-k values.

    Args:
        logits: Logits tensor [batch, num_nodes]
        k: Number of top values to keep

    Returns:
        Filtered logits with others set to -inf
    """
    if k <= 0:
        return logits

    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k, dim=-1)
    min_value = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_value, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Keep the smallest set of logits whose cumulative probability >= p.

    Args:
        logits: Logits tensor [batch, num_nodes]
        p: Cumulative probability threshold

    Returns:
        Filtered logits with others set to -inf
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # Scatter back to original ordering
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float("-inf"))


def batchify(td: TensorDict, num_repeats: int) -> TensorDict:
    """
    Repeat TensorDict for multistart.

    Args:
        td: TensorDict [batch, ...]
        num_repeats: Number of times to repeat

    Returns:
        TensorDict [batch * num_repeats, ...]
    """
    # Use interleave to keep instances together
    # Tensordict 0.3.1 might not propagate reshape(-1) to all keys correctly
    try:
        return td.repeat_interleave(num_repeats, dim=0)
    except Exception:
        # Fallback for very old versions or nested structures
        # We must set batch_size FIRST to allow item assignment of different shape
        new_batch_size = torch.Size([td.batch_size[0] * num_repeats, *td.batch_size[1:]])
        new_td = TensorDict({}, batch_size=new_batch_size, device=td.device)
        for key, val in td.items():
            if isinstance(val, torch.Tensor):
                new_td[key] = val.repeat_interleave(num_repeats, dim=0)
            elif isinstance(val, (TensorDict, dict)):
                # If it's a dict, we still need to try expanding its contents if it has tensors
                if isinstance(val, dict):
                    new_td[key] = {
                        k: (v.repeat_interleave(num_repeats, dim=0) if isinstance(v, torch.Tensor) else v)
                        for k, v in val.items()
                    }
                else:
                    # Recursively call this batchify if it's a TensorDict
                    new_td[key] = batchify(val, num_repeats)
        return new_td


def unbatchify(td: TensorDict | torch.Tensor, num_repeats: int) -> TensorDict | torch.Tensor:
    """
    Unbatchify a TensorDict/Tensor by extracting the original batch.
    Args:
        td: TensorDict/Tensor [batch * num_repeats, ...]
        num_repeats: Number of repeats
    Returns:
        Unbatchified object. If Tensor, [batch, num_repeats, ...].
        If TensorDict, [batch, ...] (takes first repeat).
    """
    if isinstance(td, torch.Tensor):
        return td.view(-1, num_repeats, *td.shape[1:])

    batch_size = td.batch_size[0] // num_repeats
    indices = torch.arange(0, batch_size * num_repeats, num_repeats, device=td.device)
    return cast(TensorDict, td[indices])


def gather_by_index(src: torch.Tensor, idx: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    Gather elements from src using indices.

    Args:
        src: Source tensor
        idx: Index tensor
        dim: Dimension to gather along

    Returns:
        Gathered tensor
    """
    idx = idx.unsqueeze(-1).expand(*idx.shape, src.shape[-1])
    return src.gather(dim, idx)


def get_log_likelihood(
    log_probs: torch.Tensor,
    actions: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    return_sum: bool = True,
) -> torch.Tensor:
    """
    Get log likelihood of selected actions.

    Args:
        log_probs: Log probabilities [batch, seq_len, num_nodes] or [batch, num_nodes].
        actions: Action indices [batch, seq_len] or [batch]. If None, assumes log_probs are already selected.
        mask: Optional mask to select only certain actions [batch, seq_len].
        return_sum: Whether to sum log probs (True) or return sequence (False).

    Returns:
        Log likelihood tensor. Shape [batch] if return_sum=True, else [batch, seq_len].
    """
    if actions is None:
        # Assume log_probs are already the selected log probabilities
        log_ll = log_probs
    # Gather log probs for selected actions
    elif log_probs.dim() == 3:
        # [batch, seq_len, num_nodes] -> gather on last dim
        log_ll = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    else:
        # [batch, num_nodes] -> gather on last dim
        log_ll = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Apply mask if provided
    if mask is not None:
        log_ll = log_ll * mask

    # Return sum or sequence
    if return_sum:
        return log_ll.sum(dim=-1) if log_ll.dim() > 1 else log_ll
    return log_ll


def modify_logits_for_top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Filter logits to keep only top-k values. Alias for top_k_filter for rl4co compatibility.

    Args:
        logits: Logits tensor [batch, num_nodes]
        top_k: Number of top values to keep

    Returns:
        Filtered logits with others set to -inf
    """
    return top_k_filter(logits, top_k)


def modify_logits_for_top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling. Alias for top_p_filter for rl4co compatibility.

    Args:
        logits: Logits tensor [batch, num_nodes]
        top_p: Cumulative probability threshold

    Returns:
        Filtered logits with others set to -inf
    """
    return top_p_filter(logits, top_p)


def segment_topk_idx(x: torch.Tensor, k: int, ids: torch.Tensor) -> torch.Tensor:
    """
    Finds the topk per segment of data x given segment ids.

    Args:
        x: Data tensor to sort [N].
        k: Number of top elements to select per segment.
        ids: Segment identifiers [N] (must be sorted).

    Returns:
        Indices of the topk elements.
    """
    assert x.dim() == 1
    assert ids.dim() == 1

    # Since we may have varying beam size per batch entry we cannot reshape to (batch_size, beam_size)
    # And use default topk along dim -1, so we have to be creative
    splits_ = torch.nonzero(ids[1:] - ids[:-1])

    if len(splits_) == 0:  # Only one group
        _, idx_topk = x.topk(min(k, x.size(0)))
        return idx_topk

    splits = torch.cat((ids.new_tensor([0]), splits_[:, 0] + 1))
    # Make a new array in which we store for each id the offset (start) of the group
    group_offsets = splits.new_zeros((int(splits.max()) + 1,))
    group_offsets[ids[splits]] = splits
    offsets = group_offsets[ids]

    # We want topk so need to sort x descending so sort -x
    idx_sorted = torch_lexsort((-(x if x.dtype != torch.uint8 else x.int()).detach(), ids))

    # Filter first k per group
    return idx_sorted[torch.arange(ids.size(0), out=ids.new()) < offsets + k]


def backtrack(parents: list[torch.Tensor], actions: list[torch.Tensor]) -> torch.Tensor:
    """
    Reconstructs action sequences by backtracking through parents.

    Args:
        parents: List of parent indices for each step.
        actions: List of actions taken at each step.

    Returns:
        Reconstructed sequences [batch_size, seq_len].
    """
    if not parents:
        return torch.empty((0, 0))

    # Now backtrack to find aligned action sequences in reversed order
    cur_parent = parents[-1]
    reversed_aligned_sequences = [actions[-1]]
    for parent, sequence in reversed(list(zip(parents[:-1], actions[:-1], strict=False))):
        reversed_aligned_sequences.append(sequence.gather(-1, cur_parent))
        cur_parent = parent.gather(-1, cur_parent)

    return torch.stack(list(reversed(reversed_aligned_sequences)), -1)


class CachedLookup:
    """
    Helper class for cached data access in beam search.

    Attributes:
        orig: Original data.
        data: Cached data.
        key: Cached key.
        current: Current data.
    """

    def __init__(self, data=None, **kwargs):
        """
        Initializes the lookup cache.

        Args:
            data: Data to cache.
            kwargs: Keyword arguments.
        """
        self.orig = kwargs
        self.data = data
        self.key = None
        self.current = None

    def __getitem__(self, key):
        """
        Retrieves data with caching.

        Args:
            key: Key to retrieve.

        Returns:
            Data.
        """
        if torch.is_tensor(key):
            if self.key is None or key.shape != self.key.shape or not torch.equal(key, self.key):
                self.key = key
                self.current = {k: v[key] for k, v in self.orig.items()}
                if self.data is not None:
                    self.current["_data"] = self.data[key]

            if self.data is not None and not self.orig:
                return self.current["_data"]  # type: ignore[index]
            return self.current

        if key in self.orig:
            return self.orig[key]
        if self.data is not None:
            return self.data[key]
        raise KeyError(key)
