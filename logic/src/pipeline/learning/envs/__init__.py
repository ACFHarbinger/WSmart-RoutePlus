"""Environments, instance generators and problem tasks.

Each problem lives in one module, ``logic/src/pipeline/learning/envs/<problem>.py``, holding its
instance generator, environment and task class. The shared machinery is in
``envs/base/`` (environment bases, ``Generator``/``RoutingGenerator``,
``BaseProblem``, resources), ``envs/common/`` (torch-free objectives, KPIs and
released specs), ``envs/core/`` and ``envs/spec.py`` (``ProblemSpec`` registry).

Nothing is imported eagerly: names resolve on first access, and ``get_env`` /
``get_generator`` import only the module of the requested problem, so importing
shared objectives or KPIs requires no neural extra.

Attributes:
    ENV_TARGETS: Environment name -> lazy import target.
    GENERATOR_TARGETS: Generator name -> lazy import target.
    GENERATOR_REGISTRY: Read-only mapping name -> generator class, imported on access.
    get_env: Build an environment by name.
    get_generator: Build an instance generator by name.

Example:
    >>> from logic.src.pipeline.learning.envs import get_env, get_generator
    >>> env = get_env("cptp", num_loc=20)
    >>> td = get_generator("ptp", num_loc=20)(batch_size=4)
"""

from collections.abc import Iterator, Mapping
from typing import Any, Dict

from logic.src.utils.registry import import_target, lazy_getattr

_PROBLEMS = ("ptp", "cptp", "tcptp", "cvrp", "tsp", "atsp", "op", "pctsp", "spctsp", "pdp", "irp", "thop")
_CLASS_PREFIX = {name: name.upper() for name in _PROBLEMS} | {"thop": "ThOP"}

ENV_TARGETS: Dict[str, str] = {
    **{name: f"logic.src.pipeline.learning.envs.{name}:{_CLASS_PREFIX[name]}Env" for name in _PROBLEMS},
    "tsp_kopt": "logic.src.pipeline.learning.envs.tsp_kopt:TSPkoptEnv",
}

# Kept for callers that inspect keys. Values are resolved on get_env.
ENV_REGISTRY = ENV_TARGETS

GENERATOR_TARGETS: Dict[str, str] = {
    **{
        name: f"logic.src.pipeline.learning.envs.{name}:{_CLASS_PREFIX[name]}Generator"
        for name in _PROBLEMS
        if name != "spctsp"
    },
    "spctsp": "logic.src.pipeline.learning.envs.pctsp:PCTSPGenerator",  # SPCTSP reuses the PCTSP generator
}

_TARGETS: Dict[str, str] = {
    "BIN_CAPACITY": "logic.src.constants.tasks:BIN_CAPACITY",
    "COST_KM": "logic.src.constants.tasks:COST_KM",
    "REVENUE_KG": "logic.src.constants.tasks:REVENUE_KG",
    "VEHICLE_CAPACITY": "logic.src.constants.tasks:VEHICLE_CAPACITY",
    "RL4COEnvBase": "logic.src.pipeline.learning.envs.base.base:RL4COEnvBase",
    "ImprovementEnvBase": "logic.src.pipeline.learning.envs.base.improvement:ImprovementEnvBase",
    "RoutingEnv": "logic.src.pipeline.learning.envs.base.routing:RoutingEnv",
    "Resource": "logic.src.pipeline.learning.envs.base.resources:Resource",
    "LoadResource": "logic.src.pipeline.learning.envs.base.resources:LoadResource",
    "ShiftTimeResource": "logic.src.pipeline.learning.envs.base.resources:ShiftTimeResource",
    "Generator": "logic.src.pipeline.learning.envs.base.generator:Generator",
    "RoutingGenerator": "logic.src.pipeline.learning.envs.base.generator:RoutingGenerator",
    "BaseProblem": "logic.src.pipeline.learning.envs.base.problem:BaseProblem",
    "ProblemSpec": "logic.src.pipeline.learning.envs.spec:ProblemSpec",
    "ProblemRegistry": "logic.src.pipeline.learning.envs.spec:ProblemRegistry",
    "PROBLEMS": "logic.src.pipeline.learning.envs.spec:PROBLEMS",
    "TSPkoptEnv": "logic.src.pipeline.learning.envs.tsp_kopt:TSPkoptEnv",
    "SPCTSPGenerator": "logic.src.pipeline.learning.envs.spctsp:SPCTSPGenerator",
    **{f"{_CLASS_PREFIX[n]}Env": f"logic.src.pipeline.learning.envs.{n}:{_CLASS_PREFIX[n]}Env" for n in _PROBLEMS},
    **{_CLASS_PREFIX[n]: f"logic.src.pipeline.learning.envs.{n}:{_CLASS_PREFIX[n]}" for n in _PROBLEMS},
    **{f"{_CLASS_PREFIX[n]}Generator": GENERATOR_TARGETS[n] for n in _PROBLEMS if n != "spctsp"},
}


class _LazyRegistry(Mapping):
    """A name -> class mapping that imports a class only when it is looked up."""

    def __init__(self, targets: Dict[str, str]) -> None:
        self._targets = targets

    def __getitem__(self, name: str) -> Any:
        return import_target(self._targets[name])

    def __iter__(self) -> Iterator[str]:
        return iter(self._targets)

    def __len__(self) -> int:
        return len(self._targets)


GENERATOR_REGISTRY: Mapping = _LazyRegistry(GENERATOR_TARGETS)


def get_env(name: str, **kwargs: Any) -> Any:
    """Build an environment by name, importing only that env module."""
    key = name.lower()
    if key not in ENV_TARGETS:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENV_TARGETS.keys())}")
    return import_target(ENV_TARGETS[key])(**kwargs)


def get_generator(name: str, **kwargs: Any) -> Any:
    """Build an instance generator by name (e.g. "ptp", "cptp", "tsp", "irp", "atsp", "cvrp").

    Raises:
        ValueError: If the generator name is not found.
    """
    if name not in GENERATOR_TARGETS:
        raise ValueError(f"Unknown generator: {name}. Available: {list(GENERATOR_TARGETS.keys())}")
    return GENERATOR_REGISTRY[name](**kwargs)


def __getattr__(name: str) -> Any:
    return lazy_getattr(globals(), _TARGETS, name)


__all__ = [
    "ENV_REGISTRY",
    "ENV_TARGETS",
    "GENERATOR_REGISTRY",
    "GENERATOR_TARGETS",
    "get_env",
    "get_generator",
    *_TARGETS,
]
