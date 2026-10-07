"""
Problem-specific utility functions.

Attributes:
    is_ptp_problem: Check if the problem is a Profitable Tour Problem (PTP) variant.
    is_tsp_problem: Check if the problem is a Traveling Salesperson Problem (TSP) variant.

Example:
    >>> from logic.src.utils.functions import is_ptp_problem, is_tsp_problem
    >>> is_ptp_problem("mvptp")
    True
    >>> is_tsp_problem("tsp")
    True
"""

from typing import Any


def is_ptp_problem(problem: Any) -> bool:
    """
    Check if the problem is a Profitable Tour Problem (PTP) variant.

    Args:
        problem: Problem instance or name string.

    Returns:
        bool: True if it's a PTP variant.
    """
    name = problem if isinstance(problem, str) else getattr(problem, "NAME", "")
    name = name.lower()
    # ptp, mvptp and tcmvptp all contain "ptp"; tcmvptp subclasses MVPTP and
    # shares the node layout and reward.
    return "ptp" in name or "pcvrp" in name


def is_tsp_problem(problem: Any) -> bool:
    """
    Check if the problem is a Traveling Salesperson Problem (TSP) variant.

    Args:
        problem: Problem instance or name string.

    Returns:
        bool: True if it's a TSP variant.
    """
    name = problem if isinstance(problem, str) else getattr(problem, "NAME", "")
    name = name.lower()
    return any(tsp_tag in name for tsp_tag in ["tsp", "atsp"])
