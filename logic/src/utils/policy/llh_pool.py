"""
Low-Level Heuristic (LLH) pool for hyper-heuristics.

Provides basic operators for route improvement and perturbation,
supporting both single-route and multi-period collection plans.

Attributes:
    Plan: Type alias for a list of routes.
    LLH: Type alias for a low-level heuristic.
    h1_greedy_move: Insert a random profitable non-mandatory node.
    h2_random_drop: Remove a random non-mandatory node.
    h3_two_opt: Standard 2-opt search.
    h4_swap_nodes: Swap two nodes in the route.
    h5_relocate_node: Relocate a node to a different position.
    h6_replace_node: Replace a node with a more profitable external node.
    h7_greedy_multi: Try to insert multiple profitable nodes.
    h8_perturb_2opt: Random double bridge swap.
    h9_worst_removal: Remove node with highest distance contribution.
    h10_greedy_shuffle: Shuffle route and 2-opt it.
    _make_or_opt: Create an OR-opt operator for a given chain length.
    _make_two_opt: Create a 2-opt operator.
    _make_three_opt: Create a 3-opt operator.
    _make_cross_day_move: Create a cross-day move operator.
    _make_cross_day_swap: Create a cross-day swap operator.
    _make_day_merge_split: Create a day merge-split operator.
    _make_day_shuffle: Create a day shuffle operator.
    _make_double_bridge: Create a double bridge operator.
    INTRA_ROUTE_POOL: List of intra-route operators.
    MULTI_PERIOD_LLH_POOL: List of multi-period operators.
    LLH_POOL: Compatibility alias for the old single-route pool.
    LLHPool: Low-Level Heuristic Pool for hyper-heuristics.

Example:
    >>> import llh_pool
    >>> llh_pool.h1_greedy_move(problem, route, rng)
    >>> llh_pool.h2_random_drop(problem, route, rng)
    >>> llh_pool.h3_two_opt(problem, route, rng)
    >>> llh_pool.h4_swap_nodes(problem, route, rng)
    >>> llh_pool.h5_relocate_node(problem, route, rng)
    >>> llh_pool.h6_replace_node(problem, route, rng)
    >>> llh_pool.h7_greedy_multi(problem, route, rng)
    >>> llh_pool.h8_perturb_2opt(problem, route, rng)
    >>> llh_pool.h9_worst_removal(problem, route, rng)
    >>> llh_pool.h10_greedy_shuffle(problem, route, rng)

Single-route heuristics live in ``llh_single_route``. Multi-period operator factories live in ``llh_multi_period``. The pools and ``LLHPool`` stay here, and every moved name is re-exported.
"""

import copy
import random
from typing import Callable, List, Tuple

import numpy as np

from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.utils.policy.wrappers import (
    greedy_day_route,
    two_opt,
)

from .llh_multi_period import (
    LLH as LLH,
)
from .llh_multi_period import (
    Plan as Plan,
)
from .llh_multi_period import (
    _make_cross_day_move as _make_cross_day_move,
)
from .llh_multi_period import (
    _make_cross_day_swap as _make_cross_day_swap,
)
from .llh_multi_period import (
    _make_day_merge_split as _make_day_merge_split,
)
from .llh_multi_period import (
    _make_day_shuffle as _make_day_shuffle,
)
from .llh_multi_period import (
    _make_double_bridge as _make_double_bridge,
)
from .llh_multi_period import (
    _make_or_opt as _make_or_opt,
)
from .llh_multi_period import (
    _make_three_opt as _make_three_opt,
)
from .llh_multi_period import (
    _make_two_opt as _make_two_opt,
)
from .llh_single_route import (
    h1_greedy_move as h1_greedy_move,
)
from .llh_single_route import (
    h2_random_drop as h2_random_drop,
)
from .llh_single_route import (
    h3_two_opt as h3_two_opt,
)
from .llh_single_route import (
    h4_swap_nodes as h4_swap_nodes,
)
from .llh_single_route import (
    h5_relocate_node as h5_relocate_node,
)
from .llh_single_route import (
    h6_replace_node as h6_replace_node,
)
from .llh_single_route import (
    h7_greedy_multi as h7_greedy_multi,
)
from .llh_single_route import (
    h8_perturb_2opt as h8_perturb_2opt,
)
from .llh_single_route import (
    h9_worst_removal as h9_worst_removal,
)
from .llh_single_route import (
    h10_greedy_shuffle as h10_greedy_shuffle,
)

# --- LLH Pools ---

INTRA_ROUTE_POOL = [
    h1_greedy_move,
    h2_random_drop,
    h3_two_opt,
    h4_swap_nodes,
    h5_relocate_node,
    h6_replace_node,
    h7_greedy_multi,
    h8_perturb_2opt,
    h9_worst_removal,
    h10_greedy_shuffle,
]

MULTI_PERIOD_LLH_POOL: List[Tuple[str, LLH]] = [
    ("or_opt_1", _make_or_opt(1)),
    ("or_opt_2", _make_or_opt(2)),
    ("or_opt_3", _make_or_opt(3)),
    ("two_opt", _make_two_opt()),
    ("three_opt", _make_three_opt()),
    ("cross_day_move", _make_cross_day_move()),
    ("cross_day_swap", _make_cross_day_swap()),
    ("day_merge_split", _make_day_merge_split()),
    ("double_bridge", _make_double_bridge()),
    ("day_shuffle", _make_day_shuffle()),
]

# Compatibility alias for the old single-route pool
LLH_POOL = INTRA_ROUTE_POOL


class LLHPool:
    """
    Low-Level Heuristic Pool for hyper-heuristics.
    Contains different perturbation/LS operators that the HH can select from.
    All LLHs take (plan, problem_context, rng) and return a new plan.

    Attributes:
        None
    """

    @staticmethod
    def llh_swap(plan: List[List[List[int]]], problem: ProblemContext, rng: random.Random) -> List[List[List[int]]]:
        """
        Swaps two random adjacent nodes in a random route.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        new_plan = copy.deepcopy(plan)
        D = problem.horizon
        d = rng.randint(0, D - 1)
        rt = new_plan[d][0] if new_plan[d] else []
        if len(rt) >= 2:
            i = rng.randint(0, len(rt) - 2)
            rt[i], rt[i + 1] = rt[i + 1], rt[i]
            new_plan[d] = [rt]
        return new_plan

    @staticmethod
    def llh_drop_add(plan: List[List[List[int]]], problem: ProblemContext, rng: random.Random) -> List[List[List[int]]]:
        """
        Drops a node and adds a new one on a random day.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        new_plan = copy.deepcopy(plan)
        D = problem.horizon
        d = rng.randint(0, D - 1)
        rt = new_plan[d][0] if new_plan[d] else []

        if rt:
            # drop non-mandatory
            cand_rem = [v for v in rt if v not in problem.mandatory]
            if cand_rem:
                rt.remove(rng.choice(cand_rem))

        # add valid
        cand_add = [v for v in range(1, len(problem.distance_matrix)) if v not in rt]
        if cand_add:
            v_add = rng.choice(cand_add)
            w_sum = sum(problem.wastes.get(n, 0.0) for n in rt)
            if w_sum + problem.wastes.get(v_add, 0.0) <= problem.capacity:
                rt.append(v_add)

        new_plan[d] = [rt]
        return new_plan

    @staticmethod
    def llh_2opt_all(plan: List[List[List[int]]], problem: ProblemContext, rng: random.Random) -> List[List[List[int]]]:
        """
        Applies 2-opt to all days.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        new_plan = []
        for d in range(problem.horizon):
            rt = plan[d][0] if plan[d] else []
            new_plan.append([two_opt(list(rt), problem.distance_matrix)])
        return new_plan

    @staticmethod
    def llh_ruin_recreate_day(
        plan: List[List[List[int]]], problem: ProblemContext, rng: random.Random
    ) -> List[List[List[int]]]:
        """
        Completely reconstructs a random day.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        new_plan = copy.deepcopy(plan)
        D = problem.horizon
        d = rng.randint(0, D - 1)

        # Advance context to day d
        cur_prob = problem
        for i in range(d):
            r = new_plan[i][0] if new_plan[i] else []
            cur_prob = cur_prob.advance(r)

        np_rng = np.random.default_rng(rng.randint(0, 2**31))
        rt = greedy_day_route(cur_prob, np_rng)
        new_plan[d] = [rt]
        return new_plan

    @classmethod
    def get_all(cls) -> List[Callable]:
        """
        Returns all LLHs in the pool.

        Returns:
            List of LLHs.
        """
        return [cls.llh_swap, cls.llh_drop_add, cls.llh_2opt_all, cls.llh_ruin_recreate_day]
