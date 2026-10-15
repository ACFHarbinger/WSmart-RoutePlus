"""Factory functions for loading problem classes."""

from __future__ import annotations

from typing import Any, Type

from logic.src.envs.tasks import CPTP, CVRP, PTP, TCPTP


def load_problem(name: str) -> Type[Any]:
    """Load a problem class by name."""
    mapping = {
        "ptp": PTP,
        "cptp": CPTP,
        "tcptp": TCPTP,
        "cvrp": CVRP,
    }
    problem = mapping.get(name.lower())
    assert problem is not None, f"unsupported problem: {name}"
    return problem
