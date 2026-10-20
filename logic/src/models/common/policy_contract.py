"""Forward contract of the released neural policy (P-24).

The Attention Model and the imitation expert are called as
``policy(td, env, strategy=..., actions=None)`` and return a ``dict``:

- ``td``: the environment state after ``env.reset``. The policy may mutate it.
- ``strategy``: ``"sampling"`` in training, ``"greedy"`` in validation, test and the
  simulator. The expert ignores it.
- ``actions`` (teacher forcing): ``[batch, steps]`` node indices to score instead of decoding.
  Decoding stops when every instance is done, so the width must cover the longest sequence;
  entries after an instance's day end are padding and are not read.

Output keys:

- ``reward`` ``[batch]``: reward of the decoded or given actions.
- ``actions`` ``[batch, steps]`` (long): visited nodes, depot 0, padded with 0 after the day end.
- ``log_likelihood`` ``[batch]``: summed log-probability of ``actions``; it carries the gradient
  in training and under teacher forcing. The expert has none.
- ``entropy`` ``[batch]``, optional: summed decoding entropy (sampling only).
- ``td``, optional: the final environment state.

Consumers read only these keys: the training step (``StepMixin.shared_step``), the imitation
loss (teacher forcing) and the simulator's neural agent (``actions``). The Attention Model's
``cost``, ``pi`` and ``log_p`` aliases serve unreleased consumers.

Attributes:
    STRATEGIES: Decoding strategies of the contract.
    check_policy_output: Validate an output against the contract.

Example:
    >>> out = policy(env.reset(td), env, strategy="sampling")
    >>> check_policy_output(out, batch_size=td.batch_size[0])"""

from __future__ import annotations

from typing import Any, Mapping

import torch

STRATEGIES = ("greedy", "sampling")


def check_policy_output(out: Mapping[str, Any], batch_size: int, needs_log_likelihood: bool = True) -> None:
    """Raise ``ValueError`` when ``out`` breaks the contract.

    Args:
        out: Policy output.
        batch_size: Number of instances in the call.
        needs_log_likelihood: Whether ``log_likelihood`` is required (False for the expert).

    Raises:
        ValueError: On a missing key or a wrong shape or dtype.
    """
    required = ["reward", "actions"] + (["log_likelihood"] if needs_log_likelihood else [])
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"policy output lacks {missing}")
    actions = out["actions"]
    if actions.dim() != 2 or actions.size(0) != batch_size or actions.dtype != torch.long:
        raise ValueError(
            f"actions must be long [batch={batch_size}, steps], got {actions.dtype} {tuple(actions.shape)}"
        )
    for key in ("reward", "log_likelihood", "entropy"):
        if key in out and tuple(out[key].shape) != (batch_size,):
            raise ValueError(f"{key} must be [batch={batch_size}], got {tuple(out[key].shape)}")
