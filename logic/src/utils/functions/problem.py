"""
Problem-specific utility functions.

Attributes:
    is_wc_problem: Check if the problem is a Waste Collection (WC) variant.
    is_vrpp_problem: Check if the problem is a Vehicle Routing Problem with Profits (VRPP) variant.
    is_tsp_problem: Check if the problem is a Traveling Salesperson Problem (TSP) variant.

Example:
    >>> from logic.src.utils.functions import is_wc_problem, is_vrpp_problem, is_tsp_problem
    >>> is_wc_problem("wcvrp")
    True
    >>> is_vrpp_problem("cvrpp")
    True
    >>> is_tsp_problem("tsp")
    True
"""

from typing import Any


def is_wc_problem(problem: Any) -> bool:
    """
    Check if the problem is a Waste Collection (WC) variant.

    Args:
        problem: Problem instance or name string.

    Returns:
        bool: True if it's a WC variant.
    """
    name = problem if isinstance(problem, str) else getattr(problem, "NAME", "")
    name = name.lower()
    return any(wc_tag in name for wc_tag in ["wcvrp", "swcvrp", "sdwcvrp"])


def is_vrpp_problem(problem: Any) -> bool:
    """
    Check if the problem is a Vehicle Routing Problem with Profits (VRPP) variant.

    Args:
        problem: Problem instance or name string.

    Returns:
        bool: True if it's a VRPP variant.
    """
    name = problem if isinstance(problem, str) else getattr(problem, "NAME", "")
    name = name.lower()
    # ctop subclasses VRPP (logic.src.envs.tasks.ctop.CTOP(VRPP)) and shares
    # its node layout/reward -- it belongs to this family even though "ctop"
    # doesn't contain the substring "vrpp".
    return any(vrpp_tag in name for vrpp_tag in ["vrpp", "cvrpp", "pcvrp"]) or name == "ctop"


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
