"""
HGS Local Search operators package.

This package provides a comprehensive suite of local search operators for VRP:
- Move operators: relocate, swap
- Route operators: 2-opt, 3-opt, 2-opt*, swap*
- Perturbation operators: perturb, kick
- Destroy operators: random, worst, cluster, shaw, string removal (SISR)
- Repair operators: greedy, regret-2, regret-k, blink insertion (SISR)
- Exchange operators: or-opt, cross-exchange, ejection chain, λ-interchange

Attributes:
    cluster_removal: Removal of clusters of nodes.
    random_removal: Removal of random nodes.
    shaw_removal: Removal of nodes based on Shaw's heuristic.
    string_removal: Removal of contiguous sequences of nodes (strings).
    worst_removal: Removal of worst nodes.
    move_or_opt: Exchange operator for or-opt.
    move_relocate: Move operator for relocating nodes.
    move_swap: Move operator for swapping nodes.
    move_2opt_intra: 2-opt intra-route operator.
    move_2opt_star: 2-opt star intra-route operator.
    move_3opt_intra: 3-opt intra-route operator.
    move_swap_star: Swap star intra-route operator.
    perturb: Perturbation operator.
    kick: Kick operator.
    greedy_insertion: Greedy insertion repair operator.
    regret_2_insertion: 2-regret insertion repair operator.
    regret_k_insertion: k-regret insertion repair operator.
    greedy_insertion_with_blinks: Greedy insertion repair operator with blinks.

Example:
    >>> from logic.src.policies.operators.destroy import random_removal
    >>> new_routes = random_removal(routes, dist_matrix)
"""

# Destroy operators
from .destroy_operators import (
    cluster_removal,
    random_removal,
    shaw_removal,
    string_removal,
    worst_removal,
)

# Exchange operators
from .exchange_operators import (
    move_or_opt,
)

# Move operators
from .move_operators import move_relocate, move_swap

# Repair operators
from .repair_operators import (
    greedy_insertion,
    greedy_insertion_with_blinks,
    regret_2_insertion,
    regret_k_insertion,
)

# Route operators
from .route_operators import (
    move_2opt_intra,
    move_2opt_star,
    move_3opt_intra,
    move_swap_star,
)

__all__ = [
    # Move
    "move_relocate",
    "move_swap",
    # Route
    "move_2opt_intra",
    "move_2opt_star",
    "move_3opt_intra",
    "move_swap_star",
    # Perturbation
    "perturb",
    "kick",
    # Destroy
    "random_removal",
    "worst_removal",
    "cluster_removal",
    "shaw_removal",
    "string_removal",
    # Repair
    "greedy_insertion",
    "regret_2_insertion",
    "regret_k_insertion",
    "greedy_insertion_with_blinks",
    # Exchange
    "move_or_opt",
]
