"""
Factory functions for loading problem classes.

Attributes:
    load_problem: Factory function to load a problem class by name.

Example:
    >>> from logic.src.utils.model.problem_factory import load_problem
    >>> problem = load_problem("vrpp")
    >>> isinstance(problem, type)
    True
"""

from __future__ import annotations

from typing import Any, Type

from logic.src.envs.problems import (
    CTOP,
    CVRPP,
    VRPP,
)


def load_problem(name: str) -> Type[Any]:
    """
    Factory function to load a problem class by name.

    Args:
        name: The problem name (e.g., 'vrpp', 'cvrpp').

    Returns:
        The problem class.

    Raises:
        AssertionError: If problem name is unsupported.
    """
    problem = {
        "vrpp": VRPP,
        "cvrpp": CVRPP,
        "ctop": CTOP,
    }.get(name)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem
