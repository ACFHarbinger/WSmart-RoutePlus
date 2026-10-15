"""Factory functions for loading problem classes."""

from __future__ import annotations

from typing import Any, Type

from logic.src.envs.tasks.cptp import CPTP
from logic.src.envs.tasks.ptp import PTP
from logic.src.envs.tasks.tcptp import TCPTP


def load_problem(name: str) -> Type[Any]:
    """Load a problem class by name."""
    problem = {
        "ptp": PTP,
        "cptp": CPTP,
        "tcptp": TCPTP,
    }.get(name, None)
    assert problem is not None, f"Problem {name} is not implemented"
    return problem
