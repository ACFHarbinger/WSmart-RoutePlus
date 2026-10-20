"""Single-route low-level heuristics h1 through h10 used by the hyper-heuristic LLH pools."""

from typing import List

import numpy as np

from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.utils.policy.wrappers import two_opt

# --- Intra-Route Operators (Single Route) ---


def h1_greedy_move(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Insert a random profitable non-mandatory node.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    nodes = set(range(1, len(problem.distance_matrix)))
    cand = list(nodes - set(route))
    if not cand:
        return route
    v = rng.choice(cand)
    if sum(problem.wastes.get(n, 0.0) for n in route) + problem.wastes.get(v, 0.0) <= problem.capacity:
        new_rt = route + [v]
        return two_opt(new_rt, problem.distance_matrix)
    return route


def h2_random_drop(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Remove a random non-mandatory node.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    cand = [v for v in route if v not in problem.mandatory]
    if not cand:
        return route
    v = rng.choice(cand)
    new_rt = [n for n in route if n != v]
    return two_opt(new_rt, problem.distance_matrix)


def h3_two_opt(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Standard 2-opt search.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    return two_opt(list(route), problem.distance_matrix)


def h4_swap_nodes(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Swap two nodes in the route.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    if len(route) < 2:
        return route
    new_rt = list(route)
    i, j = rng.choice(range(len(route)), size=2, replace=False)
    new_rt[i], new_rt[j] = new_rt[j], new_rt[i]
    return new_rt


def h5_relocate_node(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Relocate a node to a different position.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    if len(route) < 2:
        return route
    new_rt = list(route)
    v = new_rt.pop(rng.integers(len(new_rt)))
    new_rt.insert(rng.integers(len(new_rt) + 1), v)
    return new_rt


def h6_replace_node(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Replace a node with a more profitable external node.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    cand_ext = list(set(range(1, len(problem.distance_matrix))) - set(route))
    cand_in = [v for v in route if v not in problem.mandatory]
    if not cand_ext or not cand_in:
        return route
    v_in = rng.choice(cand_in)
    v_ext = rng.choice(cand_ext)
    # Check capacity
    if (
        sum(problem.wastes.get(n, 0.0) for n in route) - problem.wastes.get(v_in, 0.0) + problem.wastes.get(v_ext, 0.0)
        <= problem.capacity
    ):
        new_rt = [v_ext if n == v_in else n for n in route]
        return two_opt(new_rt, problem.distance_matrix)
    return route


def h7_greedy_multi(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Try to insert multiple profitable nodes.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    res = list(route)
    for _ in range(3):
        res = h1_greedy_move(problem, res, rng)
    return res


def h8_perturb_2opt(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Random double bridge swap.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    if len(route) < 4:
        return route
    # Simplified: just do 2 random swaps
    res = h4_swap_nodes(problem, route, rng)
    return h4_swap_nodes(problem, res, rng)


def h9_worst_removal(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Remove node with highest distance contribution.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    if len(route) < 2:
        return route
    path = [0] + route + [0]
    costs = []
    for i in range(len(route)):
        # cost added by node i: dist(prev, i) + dist(i, next) - dist(prev, next)
        c = (
            problem.distance_matrix[path[i]][path[i + 1]]
            + problem.distance_matrix[path[i + 1]][path[i + 2]]
            - problem.distance_matrix[path[i]][path[i + 2]]
        )
        # Weight by profit (low profit high cost is worst)
        p = problem.wastes.get(route[i], 1e-6)
        costs.append(c / p)

    idx = np.argmax(costs)
    if route[idx] in problem.mandatory:
        return route
    new_rt = [n for i, n in enumerate(route) if i != idx]
    return new_rt


def h10_greedy_shuffle(problem: ProblemContext, route: List[int], rng: np.random.Generator) -> List[int]:
    """
    Shuffle route and 2-opt it.

    Args:
        problem: Problem context.
        route: Route to modify.
        rng: Random number generator.

    Returns:
        Modified route.
    """
    new_rt = list(route)
    rng.shuffle(new_rt)
    return two_opt(new_rt, problem.distance_matrix)
