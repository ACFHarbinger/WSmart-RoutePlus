"""Factories for the multi-period low-level heuristics (OR-opt, 2-opt, 3-opt, cross-day moves, double bridge, day shuffle)."""

from typing import (
    Callable,
    List,
)

import numpy as np

from logic.src.interfaces.context.problem_context import ProblemContext
from logic.src.policies.helpers.operators.intra_route_local_search import (
    or_opt_route,
    three_opt_route,
    two_opt_route,
)
from logic.src.policies.helpers.operators.perturbation_shaking import (
    cross_day_move,
    cross_day_swap,
    day_merge_split,
    day_shuffle,
    double_bridge,
)

# --- Type Aliases ---
Plan = List[List[int]]
LLH = Callable[[Plan, ProblemContext, np.random.Generator], Plan]


# --- Multi-Period Operator Builders ---


def _make_or_opt(chain_len: int) -> LLH:
    """
    Create an OR-opt operator for a specific chain length.

    Args:
        chain_len: Length of the chain to optimize.

    Returns:
        OR-opt operator.
    """

    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        """
        Apply OR-opt to a random route in the plan.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        d = int(rng.integers(0, len(plan)))
        new_route = or_opt_route(plan[d], problem.distance_matrix, chain_lengths=(chain_len,))
        new_plan = list(plan)
        new_plan[d] = new_route
        return new_plan

    return _op


def _make_two_opt() -> LLH:
    """
    Create a 2-opt operator for multi-period plans.

    Returns:
        2-opt operator.
    """

    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        """
        Apply 2-opt to a random route in the plan.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        d = int(rng.integers(0, len(plan)))
        new_plan = list(plan)
        new_plan[d] = two_opt_route(plan[d], problem.distance_matrix)
        return new_plan

    return _op


def _make_three_opt() -> LLH:
    """
    Create a 3-opt operator for multi-period plans.

    Returns:
        3-opt operator.
    """

    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        """
        Apply 3-opt to a random route in the plan.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        d = int(rng.integers(0, len(plan)))
        new_plan = list(plan)
        new_plan[d] = three_opt_route(plan[d], problem.distance_matrix)
        return new_plan

    return _op


def _make_cross_day_move() -> LLH:
    """
    Create a cross-day move operator for multi-period plans.

    Returns:
        Cross-day move operator.
    """

    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        """
        Apply cross-day move to a random route in the plan.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        all_pairs = [(n, d) for d, route in enumerate(plan) for n in route]
        if not all_pairs:
            return plan
        bin_id, src_day = all_pairs[int(rng.integers(0, len(all_pairs)))]
        tgt_day = int(rng.integers(0, len(plan)))
        new_plan, _ = cross_day_move(plan, problem, bin_id, src_day, tgt_day)
        return new_plan

    return _op


def _make_cross_day_swap() -> LLH:
    """
    Create a cross-day swap operator for multi-period plans.

    Returns:
        Cross-day swap operator.
    """

    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        """
        Apply cross-day swap to a random pair of routes in the plan.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        all_pairs = [(n, d) for d, route in enumerate(plan) for n in route]
        if len(all_pairs) < 2:
            return plan
        idx = rng.choice(len(all_pairs), size=2, replace=False)
        (bin_i, day_i), (bin_j, day_j) = all_pairs[idx[0]], all_pairs[idx[1]]
        new_plan, _ = cross_day_swap(plan, problem, bin_i, day_i, bin_j, day_j)
        return new_plan

    return _op


def _make_day_merge_split() -> LLH:
    """
    Create a day merge-split operator for multi-period plans.

    Returns:
        Day merge-split operator.
    """

    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        """
        Apply day merge-split to the plan.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        new_plan, _ = day_merge_split(plan, problem, rng)
        return new_plan

    return _op


def _make_double_bridge() -> LLH:
    """
    Create a double bridge operator for multi-period plans.

    Returns:
        Double bridge operator.
    """

    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        """
        Apply double bridge to a random route in the plan.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        d = int(rng.integers(0, len(plan)))
        new_plan = list(plan)
        double_bridge(new_plan, d, rng)
        return new_plan

    return _op


def _make_day_shuffle() -> LLH:
    """
    Create a day shuffle operator for multi-period plans.

    Returns:
        Day shuffle operator.
    """

    def _op(plan: Plan, problem: ProblemContext, rng: np.random.Generator) -> Plan:
        """
        Apply day shuffle to the plan.

        Args:
            plan: Current plan.
            problem: Problem context.
            rng: Random number generator.

        Returns:
            Modified plan.
        """
        new_plan, _ = day_shuffle(plan, problem, rng, k=3)
        return new_plan

    return _op
