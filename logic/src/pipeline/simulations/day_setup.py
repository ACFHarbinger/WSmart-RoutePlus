"""Per-day inputs of the day loop: random streams and problem flags.

The environment stream is shared by every policy on the same sample and day.
The policy stream depends on a stable policy id and must not reseed the bins.
The problem flags are read from ``ProblemSpec``.
"""

import contextlib
import random
from typing import Any

import numpy as np


def shared_env_seed(seed: int, sample_id: int, day: int) -> int:
    """Seed for waste noise. Independent of which policy is running."""
    return (int(seed) + int(sample_id) * 1_000_003 + int(day)) % (2**31)


def policy_seed(seed: int, canonical_name: str, day: int) -> int:
    """Seed for a routing policy. Does not affect the shared fill stream."""
    import zlib

    name_hash = zlib.adler32(canonical_name.encode()) & 0xFFFFFFFF
    return (int(seed) + name_hash + int(day)) % (2**31)


def seed_policy_rng(seed: int) -> None:
    """Seed Python and NumPy. Seed PyTorch only when it is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def apply_shared_bin_rng(bins: Any, seed: int) -> None:
    """Reset the bin generator to the shared environment seed."""
    if bins is None:
        return
    bins.rng = np.random.default_rng(int(seed))


def per_trip_shift(context: Any) -> bool:
    """True when each trip must fit the shift (TCPTP).

    A missing spec is loaded from the released registry by problem name.
    Unreleased names are not shift-checked.
    """
    spec = getattr(context, "problem_spec", None)
    if spec is None:
        from logic.src.pipeline.rl.envs.common.released_specs import released_spec

        spec = released_spec(str(getattr(context, "problem", "") or ""))
        if spec is not None:
            with contextlib.suppress(Exception):
                context.problem_spec = spec
    return getattr(spec, "per_trip_shift", False) is True
