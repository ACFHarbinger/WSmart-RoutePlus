"""Repair operators package.

This package contains various insertion heuristics (greedy, regret-k) used in
Large Neighborhood Search (LNS) to reinsert ejected nodes back into the
fleet's routes.

Attributes:
    vectorized_greedy_repair: Reinserts nodes greedily into the first valid position that minimizes cost.
    vectorized_regret_k_repair: Reinserts nodes by choosing the position that maximizes "regret" (cost difference between best and k-th best insertions).

Example:
    None
"""
