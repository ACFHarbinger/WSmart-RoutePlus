"""Problem flags the day loop reads from ``ProblemSpec``."""

import contextlib
from typing import Any


def per_trip_shift(context: Any) -> bool:
    """True when each trip must fit the shift (TCPTP).

    A missing spec is loaded from the released registry by problem name.
    Unreleased names are not shift-checked.
    """
    spec = getattr(context, "problem_spec", None)
    if spec is None:
        from logic.src.envs.tasks.common.released_specs import released_spec

        spec = released_spec(str(getattr(context, "problem", "") or ""))
        if spec is not None:
            with contextlib.suppress(Exception):
                context.problem_spec = spec
    return getattr(spec, "per_trip_shift", False) is True
