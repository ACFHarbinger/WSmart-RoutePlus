"""Random streams for one simulation day.

The environment stream is shared by every policy on the same sample and day.
The policy stream depends on a stable policy id and must not reseed the bins.
"""

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
