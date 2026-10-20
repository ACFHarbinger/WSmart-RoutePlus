"""Batched destroy-repair low-level heuristics (``search_heuristics.destroy_repair_llh``).

Attributes:
    LLH_NAMES: The five composites in the simulator's order.
    apply_llh: Apply one composite to a batch.
"""

from __future__ import annotations

from typing import Optional

import torch

from . import routes as rs
from .vrpp import VRPPBatch

LLH_NAMES = (
    "random_greedy",
    "worst_regret_2",
    "cluster_greedy",
    "worst_greedy",
    "random_regret_2",
)


def _pending(S: torch.Tensor, removed: torch.Tensor, inst: VRPPBatch, expand_pool: bool) -> torch.Tensor:
    pending = removed.clone()
    if expand_pool:
        pending = pending | ~rs.visited(S, inst.dist.size(1))
    pending[:, 0] = False
    return pending


def apply_llh(
    idx: int,
    S: torch.Tensor,
    n: torch.Tensor,
    inst: VRPPBatch,
    gen: Optional[torch.Generator],
    profit_aware: bool,
    expand_pool: bool = True,
) -> torch.Tensor:
    """Apply the low-level heuristic ``LLH_NAMES[idx]`` with ``n`` removals per row.

    Args:
        idx: Heuristic index.
        S: Route sequences.
        n: Removal counts ``[B]``.
        inst: Instances.
        gen: Torch generator.
        profit_aware: Use the profit-aware removal/insertion flavours.
        expand_pool: Repairs may insert any unvisited bin.

    Returns:
        torch.Tensor: New route sequences.
    """
    name = LLH_NAMES[idx]
    if name.startswith("random"):
        P, pos = rs.random_removal(S, n, gen)
        removed = torch.zeros(S.size(0), inst.dist.size(1), dtype=torch.bool, device=S.device)
        removed.scatter_(1, torch.where(pos, S, torch.zeros_like(S)), True)
        removed[:, 0] = False
    elif name.startswith("worst"):
        P, removed = rs.worst_removal(S, n, inst, 1.0, gen, profit_aware)
    else:
        P, removed = rs.cluster_removal(S, n, inst, gen)
    pending = _pending(P, removed, inst, expand_pool)
    if name.endswith("regret_2"):
        return rs.regret_insertion(P, pending, inst, 2, profit_aware)
    return rs.greedy_insertion(P, pending, inst, profit_aware)
