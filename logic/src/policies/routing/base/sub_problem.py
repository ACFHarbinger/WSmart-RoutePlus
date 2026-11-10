"""Sub-problem value object and solver protocol for routing engines (P-30).

The 1.0.0 routing consolidation bundles the per-solver positional arguments
(distance matrix, wastes, capacity, revenue, cost unit, mandatory nodes and
the optional simulator extras) into a single :class:`SubProblem` value object,
and gives every routing solver one constructor contract:

    solver = MySolver(problem: SubProblem, cfg: MyConfig)
    routes, profit, cost = solver.solve()

``run_*`` dispatchers keep their legacy role as engine-selection shims but
take the same ``(problem, cfg)`` pair, so adapters, stages, benchmarks and
tests all speak one vocabulary.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Type, runtime_checkable

import numpy as np
from omegaconf import DictConfig, OmegaConf

__all__ = ["SubProblem", "SolveResult", "Solver", "build_subproblem", "coerce_config"]


class SolveResult(NamedTuple):
    """Standard routing solver result.

    A ``NamedTuple`` so existing ``routes, profit, cost = solver.solve()``
    unpacking keeps working unchanged.
    """

    routes: List[List[int]]
    profit: float
    cost: float


@runtime_checkable
class Solver(Protocol):
    """Routing solver contract: build with a sub-problem + typed config, solve once."""

    def __init__(self, problem: "SubProblem", cfg: Any = None, **kwargs: Any) -> None:
        ...

    def solve(self) -> SolveResult:
        ...


class SubProblem:
    """The extracted routing sub-problem handed to every routing solver.

    Attributes mirror what ``BaseRoutingPolicy._run_solver`` used to receive
    as positional arguments, plus the optional simulator extras that special
    engines (BPC, SWC-TCF, HGS) previously pulled out of ``**kwargs``.
    """

    __slots__ = (
        "dist_matrix",
        "wastes",
        "capacity",
        "revenue",
        "cost_unit",
        "mandatory_nodes",
        "subset_indices",
        "x_coords",
        "y_coords",
        "values",
        "n_vehicles",
        "env",
        "node_coords",
        "recorder",
        "dual_values",
        "trace",
        "trace_parent_id",
    )

    def __init__(
        self,
        dist_matrix: np.ndarray,
        wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        mandatory_nodes: Optional[List[int]] = None,
        subset_indices: Optional[List[int]] = None,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        values: Optional[Dict[str, Any]] = None,
        n_vehicles: Optional[int] = None,
        env: Any = None,
        node_coords: Optional[np.ndarray] = None,
        recorder: Any = None,
        dual_values: Any = None,
        trace: Any = None,
        trace_parent_id: Optional[str] = None,
    ) -> None:
        self.dist_matrix = dist_matrix
        self.wastes = wastes
        self.capacity = capacity
        self.revenue = revenue
        self.cost_unit = cost_unit
        self.mandatory_nodes = list(mandatory_nodes) if mandatory_nodes is not None else []
        self.subset_indices = subset_indices
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.values = values
        self.n_vehicles = n_vehicles
        self.env = env
        self.node_coords = node_coords
        self.recorder = recorder
        self.dual_values = dual_values
        self.trace = trace
        self.trace_parent_id = trace_parent_id

    def __repr__(self) -> str:
        n = 0 if self.dist_matrix is None else len(self.dist_matrix)
        return (
            f"SubProblem(n={n}, mandatory={len(self.mandatory_nodes)}, "
            f"capacity={self.capacity}, revenue={self.revenue}, cost_unit={self.cost_unit})"
        )


def build_subproblem(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    revenue: float,
    cost_unit: float,
    mandatory_nodes: Optional[List[int]] = None,
    **extras: Any,
) -> SubProblem:
    """Build a :class:`SubProblem` from the legacy 5-tuple plus optional extras.

    Extras are forwarded as keyword arguments (``subset_indices``,
    ``x_coords``, ``y_coords``, ``values``, ``n_vehicles``, ``env``,
    ``node_coords``, ``recorder``, ``dual_values``); unknown extras are
    ignored so callers can pass a broad kwargs dictionary.
    """
    allowed = {
        "subset_indices",
        "x_coords",
        "y_coords",
        "values",
        "n_vehicles",
        "env",
        "node_coords",
        "recorder",
        "dual_values",
        "trace",
        "trace_parent_id",
    }
    filtered = {key: value for key, value in extras.items() if key in allowed and value is not None}
    return SubProblem(
        dist_matrix=dist_matrix,
        wastes=wastes,
        capacity=capacity,
        revenue=revenue,
        cost_unit=cost_unit,
        mandatory_nodes=mandatory_nodes,
        **filtered,
    )


def coerce_config(cfg: Any, config_cls: Type) -> Any:
    """Lenient engine-level config coercion.

    Mirrors the legacy routing constructor contracts: dataclass instances pass
    through, mappings (dicts and Hydra ``DictConfig`` sections) and arbitrary objects
    are filtered onto the dataclass fields, and None yields the dataclass defaults.
    Engines tolerate; adapters load strictly.

    Args:
        cfg: Target configuration dataclass instance, dict, arbitrary object, or None.
        config_cls: The target configuration dataclass.

    Returns:
        An instance of ``config_cls``.
    """
    if cfg is None:
        return config_cls()
    if isinstance(cfg, config_cls):
        return cfg
    valid_keys = {f.name for f in dataclasses.fields(config_cls)}
    if isinstance(cfg, DictConfig):
        # A nested Hydra section: read its values (vars() would only see OmegaConf's internals).
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, Mapping):
        return config_cls(**{k: v for k, v in cfg.items() if k in valid_keys})
    if hasattr(cfg, "__dict__"):
        return config_cls(**{k: v for k, v in vars(cfg).items() if k in valid_keys})
    return cfg

