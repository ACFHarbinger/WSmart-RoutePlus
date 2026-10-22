"""Factory functions for loading problem classes."""

from __future__ import annotations

from typing import Any, Type

# Problem name -> task class in logic.src.pipeline.learning.envs. Classes import on request,
# so a tree without a problem's module (the release ships PTP, CPTP, CPTPTC and CVRP) loads
# the others.
_PROBLEM_CLASSES = {
    "ptp": "PTP",
    "cptp": "CPTP",
    "cptptc": "CPTPTC",
    "cptptwtc": "CPTPTWTC",
    "cvrp": "CVRP",
}


def load_problem(name: str) -> Type[Any]:
    """Load a problem class by name."""
    class_name = _PROBLEM_CLASSES.get(name.lower())
    assert class_name is not None, f"unsupported problem: {name}"
    from logic.src.pipeline.learning import envs

    return getattr(envs, class_name)
