"""
Hyper-Heuristic Operators Module.

This module provides wrapper functions for local search operators
that can be called as standalone moves in a hyper-heuristic context.
Each operator attempts to improve a solution and returns the modified routes.

Attributes:
    HYPER_OPERATORS (list): List of operator functions.
    OPERATOR_NAMES (list): List of operator names.
    HyperOperatorContext: Context object that provides the interface expected by move operators.
    apply_2opt_intra: Wrapper for 2-opt intra-route move.
    apply_3opt_intra: Wrapper for 3-opt intra-route move.
    apply_2opt_star: Wrapper for 2-opt* inter-route move.
    apply_swap: Wrapper for swap move between two nodes.
    apply_swap_star: Wrapper for SWAP* inter-route move.
    apply_relocate: Wrapper for relocate move.
    apply_kick: Wrapper for kick operator.
    apply_perturb: Wrapper for perturbation operator.
    apply_random_removal: Wrapper for random removal operator.
    apply_shaw_removal: Wrapper for shaw removal operator.
    apply_string_removal: Wrapper for string removal operator.

Example:
    >>> from logic.src.policies.routing.hyper_heuristics.ant_colony_optimization_hyper_heuristic.hyper_operators import apply_2opt
    >>> improved = apply_2opt(context)

:class:`HyperOperatorContext` lives in ``hyper_operator_context.py`` and is re-exported here.
"""

import logging
from typing import Callable, Dict, Optional

from logic.src.pipeline.learning.envs.common.expand_pool import current_expand_pool
from logic.src.policies.helpers.operators import (
    greedy_insertion,
    greedy_profit_insertion,
    kick,
    kick_profit,
    move_2opt_intra,
    move_2opt_star,
    move_3opt_intra,
    move_relocate,
    move_swap,
    move_swap_star,
    perturb,
    random_removal,
    shaw_profit_removal,
    shaw_removal,
    string_removal,
)

from .hyper_operator_context import HyperOperatorContext as HyperOperatorContext

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Wrapper Operators
# -----------------------------------------------------------------------------


def apply_2opt_intra(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply 2-opt intra-route move to a random route segment.

    Args:
        ctx: Context containing problem data and structures.
        max_attempts: Maximum number of attempts. Defaults to 50.

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = ctx.rng.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if r_u == r_v and move_2opt_intra(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_3opt_intra(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply 3-opt intra-route move to a random route segment.

    Args:
        ctx: Context containing problem data and structures.
        max_attempts: Maximum number of attempts. Defaults to 50.

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = ctx.rng.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if r_u == r_v and move_3opt_intra(ctx, u, v, r_u, p_u, r_v, p_v, ctx.rng):
                improved = True
                break
        if improved:
            break
    return improved


def apply_2opt_star(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply 2-opt* inter-route move.

    Args:
        ctx: Context containing problem data and structures.
        max_attempts: Maximum number of attempts. Defaults to 50.

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = ctx.rng.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if r_u != r_v and move_2opt_star(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_swap(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply swap move between two nodes.

    Args:
        ctx: Context containing problem data and structures.
        max_attempts: Maximum number of attempts. Defaults to 50.

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = ctx.rng.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if move_swap(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_swap_star(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply SWAP* inter-route move.

    Args:
        ctx: Context containing problem data and structures.
        max_attempts: Maximum number of attempts. Defaults to 50.

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = ctx.rng.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if r_u != r_v and move_swap_star(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_relocate(ctx: HyperOperatorContext, max_attempts: int = 50) -> bool:
    """Apply relocate move.

    Args:
        ctx: Context containing problem data and structures.
        max_attempts: Maximum number of attempts. Defaults to 50.

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    improved = False
    for _ in range(max_attempts):
        nodes = [n for n in ctx.neighbors if n in ctx.node_map]
        if not nodes:
            break
        u = ctx.rng.choice(nodes)
        u_loc = ctx.node_map.get(u)
        if not u_loc:
            continue
        r_u, p_u = u_loc

        for v in ctx.neighbors.get(u, []):
            v_loc = ctx.node_map.get(v)
            if not v_loc:
                continue
            r_v, p_v = v_loc
            if move_relocate(ctx, u, v, r_u, p_u, r_v, p_v):
                improved = True
                break
        if improved:
            break
    return improved


def apply_perturb(ctx: HyperOperatorContext, k: int = 3) -> bool:
    """Wrapper for perturbation operator.

    Args:
        ctx: Context containing problem data and structures.
        k: Number of nodes to perturb. Defaults to 3.

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    return perturb(ctx, k, rng=ctx.rng)


def apply_kick(ctx: HyperOperatorContext, destroy_ratio: float = 0.2) -> bool:
    """Wrapper for kick operator.

    Args:
        ctx: Context containing problem data and structures.
        destroy_ratio: Ratio of nodes to remove. Defaults to 0.2.

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    if ctx.profit_aware_operators:
        return kick_profit(ctx, destroy_ratio, bias=2.0, rng=ctx.rng)
    return kick(ctx, destroy_ratio, rng=ctx.rng)


def apply_shaw_removal(ctx: HyperOperatorContext, n: Optional[int] = None) -> bool:
    """Wrapper for shaw removal followed by greedy insertion.

    Args:
        ctx: Context containing problem data and structures.
        n: Number of nodes to remove. Defaults to max(1, len(ctx.node_map) // 5).

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    n_remove = n if n is not None else max(1, len(ctx.node_map) // 5)
    try:
        if ctx.profit_aware_operators:
            partial, removed = shaw_profit_removal(ctx.routes, n_remove, ctx.d, ctx.waste, ctx.R, ctx.C, rng=ctx.rng)
        else:
            partial, removed = shaw_removal(ctx.routes, n_remove, ctx.d, rng=ctx.rng)

        if ctx.profit_aware_operators:
            ctx.routes = greedy_profit_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                R=ctx.R,
                C=ctx.C,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=current_expand_pool(),
            )
        else:
            ctx.routes = greedy_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=current_expand_pool(),
            )
        ctx._build_structures()
        return True
    except (ValueError, IndexError, KeyError) as exc:
        logger.warning("hyper operator degraded to no-op: %s", exc)
        return False


def apply_string_removal(ctx: HyperOperatorContext, n: Optional[int] = None) -> bool:
    """Wrapper for string removal followed by greedy insertion.

    Args:
        ctx: Context containing problem data and structures.
        n: Number of nodes to remove. Defaults to max(1, len(ctx.node_map) // 5).

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    n_remove = n if n is not None else max(1, len(ctx.node_map) // 5)
    try:
        partial, removed = string_removal(ctx.routes, n_remove, ctx.d, rng=ctx.rng)
        if ctx.profit_aware_operators:
            ctx.routes = greedy_profit_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                R=ctx.R,
                C=ctx.C,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=current_expand_pool(),
            )
        else:
            ctx.routes = greedy_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=current_expand_pool(),
            )
        ctx._build_structures()
        return True
    except (ValueError, IndexError, KeyError) as exc:
        logger.warning("hyper operator degraded to no-op: %s", exc)
        return False


def apply_random_removal(ctx: HyperOperatorContext, n: Optional[int] = None) -> bool:
    """Wrapper for random removal followed by greedy insertion.

    Args:
        ctx: Context containing problem data and structures.
        n: Number of nodes to remove. Defaults to max(1, len(ctx.node_map) // 5).

    Returns:
        bool: True if an improving move was made, False otherwise.
    """
    n_remove = n if n is not None else max(1, len(ctx.node_map) // 5)
    try:
        partial, removed = random_removal(ctx.routes, n_remove, rng=ctx.rng)
        if ctx.profit_aware_operators:
            ctx.routes = greedy_profit_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                R=ctx.R,
                C=ctx.C,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=current_expand_pool(),
            )
        else:
            ctx.routes = greedy_insertion(
                partial,
                removed,
                ctx.d,
                ctx.waste,
                ctx.Q,
                mandatory_nodes=list(ctx.mandatory_nodes),
                expand_pool=current_expand_pool(),
            )
        ctx._build_structures()
        return True
    except (ValueError, IndexError, KeyError) as exc:
        logger.warning("hyper operator degraded to no-op: %s", exc)
        return False


# Registry of available operators
HYPER_OPERATORS: Dict[str, Callable[[HyperOperatorContext], bool]] = {
    "2opt_intra": apply_2opt_intra,
    "3opt_intra": apply_3opt_intra,
    "2opt_star": apply_2opt_star,
    "swap": apply_swap,
    "swap_star": apply_swap_star,
    "relocate": apply_relocate,
    "perturb": apply_perturb,
    "kick": apply_kick,
    "shaw_removal": apply_shaw_removal,
    "string_removal": apply_string_removal,
    "random_removal": apply_random_removal,
}

OPERATOR_NAMES = list(HYPER_OPERATORS.keys())
