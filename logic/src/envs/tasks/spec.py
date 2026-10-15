"""Problem specification and registry.

This module provides ``ProblemSpec``, the single declarative source of truth
for problem definitions (objective, constraints, metrics, features, and env loading),
decoupling problem semantics from string comparisons across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Type

from logic.src.utils.registry import import_target


@dataclass(frozen=True)
class ProblemSpec:
    """Declarative specification of a routing problem.

    Attributes:
        name: Canonical problem identifier (e.g. "cptp", "ptp", "cvrp", "tcptp").
        objective: Pure economic / cost objective function.
        constraints: Tuple of declarative Resource constraints.
        repair: Optional repair operator for invalid/overweight tours.
        metrics: Tuple of reported KPI metric names.
        env_target: Lazy import target string for the RL environment class
            (e.g. "logic.src.envs.tasks.cptp:CPTPEnv").
        features: Optional feature specification for embedding projections.
        expand_pool: Whether unvisited candidate pools may expand dynamically (True for PTP/CPTP/TCPTP, False for CVRP).
    """

    name: str
    objective: Callable[..., float]
    constraints: Tuple[Any, ...] = ()
    repair: Optional[Callable[[Any, Any], Any]] = None
    metrics: Tuple[str, ...] = ()
    env_target: str = ""
    features: Optional[Any] = None
    expand_pool: bool = True

    def get_env_class(self) -> Type[Any]:
        """Lazily load the environment class for this problem."""
        if not self.env_target:
            raise ValueError(f"No env_target specified for ProblemSpec '{self.name}'.")
        return import_target(self.env_target)

    def get_env(self, **kwargs: Any) -> Any:
        """Instantiate the environment for this problem with keyword arguments."""
        env_cls = self.get_env_class()
        return env_cls(**kwargs)


class ProblemRegistry:
    """Registry mapping problem names to ProblemSpec instances."""

    def __init__(self) -> None:
        self._specs: Dict[str, ProblemSpec] = {}

    def register(self, spec: ProblemSpec) -> None:
        """Register a problem specification."""
        self._specs[spec.name.lower()] = spec

    def get(self, name: str) -> ProblemSpec:
        """Retrieve a problem specification by name."""
        key = name.lower()
        if key not in self._specs:
            raise ValueError(
                f"Unknown problem specification: {name!r}. Available: {list(self._specs.keys())}"
            )
        return self._specs[key]

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._specs

    def __getitem__(self, name: str) -> ProblemSpec:
        return self.get(name)

    def keys(self) -> list[str]:
        return list(self._specs.keys())

    def values(self) -> list[ProblemSpec]:
        return list(self._specs.values())

    def items(self) -> list[tuple[str, ProblemSpec]]:
        return list(self._specs.items())


PROBLEMS = ProblemRegistry()

__all__ = ["ProblemSpec", "PROBLEMS", "ProblemRegistry"]
