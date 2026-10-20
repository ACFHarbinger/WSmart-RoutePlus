"""
Base Routing Policy Module.

Provides a template base class for routing policies, extracting common
functionality like parameter loading, subset matrix creation, and tour mapping.

Attributes:
    BaseRoutingPolicy: Base class for all routing policies.
    _structured_load: Load a policy mapping into a typed config dataclass.

Example:
    >>> from logic.src.policies.routing.base.base_routing_policy import BaseRoutingPolicy
    >>> # subclass and implement _run_solver

The ``execute`` template and its subset-problem, tour-mapping, cost and logging helpers
live in ``routing_execution.py`` (:class:`RoutingExecutionMixin`).
"""

from abc import abstractmethod
from dataclasses import asdict, fields, is_dataclass, replace
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from logic.src.interfaces.routing_algorithm import IRoutingAlgorithm
from logic.src.pipeline.simulations.repository import load_area_and_waste_type_params
from logic.src.policies.routing.base.routing_execution import RoutingExecutionMixin
from logic.src.tracking.viz_mixin import PolicyVizMixin

_STAGE_KEYS = frozenset({"packing", "route_improvement", "acceptance_criteria", "acceptance_criterion"})
_RUNTIME_KEYS = _STAGE_KEYS | {"seed"}
_ENGINE_KEYS = ("gurobi", "ortools", "custom")


def _unwrap_engine_section(source: Any) -> Dict[str, Any]:
    """Unwrap a policy-name or engine wrapper into a plain parameter mapping."""
    if isinstance(source, (ListConfig, list)):
        raise TypeError("policy YAML must be a mapping after P-13; run tools/convert_policy_yaml.py")
    if isinstance(source, (DictConfig, ListConfig)) or OmegaConf.is_config(source):
        source = OmegaConf.to_container(source, resolve=True)
    if not isinstance(source, dict):
        return {}
    section = dict(source)
    while len(section) == 1:
        key = next(iter(section))
        if key in _STAGE_KEYS or key in _ENGINE_KEYS:
            break
        inner = section[key]
        if isinstance(inner, dict):
            section = inner
            continue
        if isinstance(inner, list):
            raise TypeError("policy YAML must be a mapping after P-13; run tools/convert_policy_yaml.py")
        break
    for engine in _ENGINE_KEYS:
        inner = section.get(engine)
        if isinstance(inner, dict):
            merged = {key: value for key, value in section.items() if key != engine}
            merged.update(inner)
            section = merged
            break
    return section


def _structured_load(config_cls: Type, mapping: Dict[str, Any]) -> Any:
    """Load ``mapping`` into ``config_cls``; unknown keys fail."""
    core = {key: value for key, value in mapping.items() if key not in _RUNTIME_KEYS}
    known = {item.name for item in fields(config_cls)}
    unknown = sorted(set(core) - known)
    if unknown:
        raise ValueError(f"unknown or invalid keys for {config_cls.__name__}: {unknown}")
    filtered = {key: value for key, value in core.items() if key in known}

    def with_acceptance(config):
        if "acceptance_criterion" in known and "acceptance_criterion" in mapping:
            from logic.src.configs.policies.other.acceptance_criteria import AcceptanceConfig

            spec = mapping["acceptance_criterion"]
            if isinstance(spec, (dict, DictConfig)):
                spec = AcceptanceConfig(method=spec.get("method", "oi"), params=spec.get("params", {}))
            return replace(config, acceptance_criterion=spec)
        return config

    try:
        schema = OmegaConf.structured(config_cls)
    except Exception:
        # Some declarations use types OmegaConf cannot schema. Only schema
        # construction may fall back; invalid user values must never bypass validation.
        return with_acceptance(config_cls(**filtered))
    OmegaConf.set_struct(schema, True)
    for item in fields(config_cls):
        node = schema.get(item.name)
        annotation = str(item.type)
        if isinstance(node, DictConfig) and (annotation == "<class 'dict'>" or annotation.startswith("typing.Dict")):
            OmegaConf.set_struct(node, False)
    merged = OmegaConf.merge(schema, OmegaConf.create(_plain(filtered)))
    return with_acceptance(OmegaConf.to_object(merged))


def _plain(value: Any) -> Any:
    """Plain dicts and lists: OmegaConf rejects dict subclasses such as the simulator's _AttrDict."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class BaseRoutingPolicy(RoutingExecutionMixin, PolicyVizMixin, IRoutingAlgorithm):
    """
    Base class for routing policies with common utilities.

    Subclasses implement `_run_solver()` and optionally `_get_config_key()`
    and `_config_class()`.

    The `execute()` method provides a template pattern handling:
        1. Mandatory validation
        2. Parameter loading
        3. Subset problem creation
        4. Solver invocation (delegated to subclass)
        5. Tour mapping back to global IDs
        6. Cost computation

    Context Inputs (from kwargs):
        mandatory: List of 1-based bin IDs to visit
        bins: Bins state object with `.c` fill levels
        distance_matrix: Full NxN distance matrix
        area: Geographic area name
        waste_type: Waste type (e.g., "plastic")
        config: Policy configuration dict

    Context Outputs:
        Tuple[List[int], float, Any]: (tour, cost, extra_output)

    Attributes:
        config: Optional configuration dataclass for this policy.
    """

    def __init__(self, config: Any = None):
        """Initialize policy with optional config dataclass or raw dict.

        Args:
            config: Dataclass config, raw dict from YAML, or None.
                    If a raw dict is passed and the subclass defines
                    ``_config_class()``, it will be parsed into the
                    typed dataclass automatically.
        """
        if config is not None and isinstance(config, dict):
            self._config, self._seed = self._build_config(config)
        else:
            self._config = config
            self._seed = getattr(config, "seed", 42) if config is not None else 42

    @property
    def config(self) -> Any:
        """Return the policy configuration dataclass.

        Returns:
            The configuration dataclass instance (e.g., HGSConfig).
        """
        return self._config

    @classmethod
    def _config_class(cls) -> Optional[Type]:
        """Return the dataclass type for this policy's config.

        Override in subclasses to enable automatic config parsing.

        Returns:
            Dataclass type (e.g., ALNSConfig) or None if no typed config is used.
        """
        return None

    @classmethod
    def _build_config(cls, raw_config: Dict[str, Any]) -> Any:
        """Build a typed config dataclass from a raw config dict.

        Extracts the policy-specific section from the raw config (using
        ``_get_config_key()``), flattens nested structures, and constructs
        the typed dataclass (from ``_config_class()``).

        Args:
            raw_config: Raw config dict (may be nested with policy key at top level).

        Returns:
            Tuple of (Typed config dataclass, seed) or (None, None).
        """

        config_cls = cls._config_class()
        if config_cls is None:
            return None, None

        valid_fields = {f.name for f in fields(config_cls)}

        # Extract policy-specific section
        config_key = cls._get_config_key(cls)  # type: ignore[arg-type]
        policy_section = raw_config.get(config_key)
        if policy_section is None:
            # The simulator keys a policy's section by its test_sim entry name (e.g. 'aco_hh',
            # 'sans_og_a'), which need not match config_key; use that single section rather
            # than silently falling back to the dataclass defaults.
            sections = [
                v
                for k, v in raw_config.items()
                # Stage keys (e.g. a flat config's `acceptance_criteria` list) are not policy sections.
                if k not in _RUNTIME_KEYS
                and k not in valid_fields
                and isinstance(v, (dict, list, DictConfig, ListConfig))
            ]
            policy_section = sections[0] if len(sections) == 1 else raw_config

        flat = _unwrap_engine_section(policy_section)

        # Ensure seed from the root config reaches flat if missing
        if "seed" not in flat and "seed" in raw_config:
            flat["seed"] = raw_config["seed"]

        cfg = _structured_load(config_cls, flat)
        for key in ("seed",):
            if key in flat and hasattr(cfg, key):
                setattr(cfg, key, flat[key])
        return cfg, flat.get("seed")

    def _get_config_key(self) -> str:
        """Return the config key for this policy (e.g., 'alns', 'bcp', 'hgs').

        Override in subclasses.

        Returns:
            String key used to look up this policy's section in the global config.
        """
        return "default"

    def _effective_config(self, values: Dict[str, Any]) -> Any:
        """Materialize runtime solver values into the typed config dataclass.

        ``values`` (from ``_load_area_params``) is ``asdict(self._config)``
        overlaid with per-call area overrides and the simulator seed. Filtering
        it back onto the config fields gives the solver a single typed source of
        truth (P-10): the solver reads ``cfg.field`` instead of rebuilding a
        parallel params object with ``getattr`` fallbacks.

        Args:
            values: Merged solver configuration dict from ``_load_area_params``.

        Returns:
            A config dataclass instance of ``self._config_class()``, or the
            original ``values`` dict when the policy has no typed config class.
        """
        config_cls = self._config_class()
        if config_cls is None:
            return values
        known = {f.name for f in fields(config_cls)}
        filtered = {key: value for key, value in values.items() if key in known}
        cfg = _structured_load(config_cls, filtered)
        # _structured_load drops _RUNTIME_KEYS (seed, stage file-refs);
        # the solver-facing ones are applied explicitly here.
        for key in ("seed",):
            if key in values:
                setattr(cfg, key, values[key])
        return cfg

    def _validate_mandatory(self, mandatory: Optional[List[int]]) -> Optional[Tuple[List[int], float, float]]:
        """
        Validate mandatory input. Returns early result if empty.

        Args:
            mandatory: List of bin IDs to visit.

        Returns:
            Optional tuple containing (empty_tour, 0.0) if no targets, else None.
        """

        if not mandatory:
            return [0, 0], 0.0, 0.0
        return None

    def _load_area_params(
        self, area: str, waste_type: str, config: Dict[str, Any]
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Load area-specific parameters (Q, R, C) and merge with config overrides.

        Uses the typed config dataclass (``self._config``) when available,
        falling back to dict-based extraction for backward compatibility.

        Args:
            area: Geographic area name.
            waste_type: Waste type string.
            config: Policy configuration dict (from context).

        Returns:
            Tuple of (capacity, revenue_scaled, cost_unit, merged_values_dict).
        """

        Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)

        # Build the policy config dict from the typed config or raw dict
        if self._config is not None and is_dataclass(self._config):
            # Typed config takes priority, then overlay any runtime overrides
            config_key = self._get_config_key()
            runtime_overrides = config.get(config_key, config)  # Handle case where config IS the policy section
            if hasattr(runtime_overrides, "items"):
                runtime_overrides = _unwrap_engine_section(runtime_overrides)
            elif isinstance(runtime_overrides, list):
                raise TypeError("policy YAML must be a mapping after P-13; run tools/convert_policy_yaml.py")
            else:
                runtime_overrides = {}
            policy_config = {**asdict(self._config), **runtime_overrides}
            if self._seed is not None and policy_config.get("seed") is None:
                policy_config["seed"] = self._seed
        else:
            # Legacy path: extract from raw config dict
            config_key = self._get_config_key()
            policy_section = config.get(config_key, config)
            if hasattr(policy_section, "items"):
                policy_config = _unwrap_engine_section(policy_section)
            elif isinstance(policy_section, list):
                raise TypeError("policy YAML must be a mapping after P-13; run tools/convert_policy_yaml.py")
            else:
                policy_config = {}
            # Ensure seed from context also makes it into legacy path if missing
            if policy_config.get("seed") is None and config.get("seed") is not None:
                policy_config["seed"] = config["seed"]

        capacity = policy_config.get("capacity", Q)
        revenue_kg = policy_config.get("revenue", R)
        cost_unit = policy_config.get("cost_unit", C)
        density = policy_config.get("density", B)
        bin_volume = policy_config.get("bin_volume", V)

        # Scale revenue to euro per percent for VRPP/Classical solvers
        # revenue_kg (€/kg) * density (kg/L) * bin_volume (L) / 100 = € per 1% fill
        revenue_scaled = revenue_kg * (density * bin_volume / 100.0)

        shift_hours = policy_config.get("shift_hours")
        avg_speed_kmh = policy_config.get("avg_speed_kmh")
        service_time_h = policy_config.get("service_time_h")
        if shift_hours is None or avg_speed_kmh is None or service_time_h is None:
            from logic.src.pipeline.simulations.repository import load_temporal_params

            t_shift, t_speed, t_service = load_temporal_params()
            shift_hours = shift_hours if shift_hours is not None else t_shift
            avg_speed_kmh = avg_speed_kmh if avg_speed_kmh is not None else t_speed
            service_time_h = service_time_h if service_time_h is not None else t_service

        values: Dict[str, Any] = {
            "Q": capacity,
            "R": revenue_scaled,
            "C": cost_unit,
            "B": density,
            "V": bin_volume,
            "shift_hours": float(shift_hours),
            "avg_speed_kmh": float(avg_speed_kmh),
            "service_time_h": float(service_time_h),
        }
        # Merge all policy config params into values for solver consumption
        values.update(policy_config)

        return capacity, revenue_scaled, cost_unit, values

    @abstractmethod
    def _run_solver(
        self,
        sub_dist_matrix: np.ndarray,
        sub_wastes: Dict[int, float],
        capacity: float,
        revenue: float,
        cost_unit: float,
        values: Dict[str, Any],
        mandatory_nodes: List[int],
        **kwargs: Any,
    ) -> Tuple[List[List[int]], float, float]:
        """
        Run the specific solver for this policy.

        Args:
            sub_dist_matrix: Subset distance matrix.
            sub_wastes: Local wastes dict.
            capacity: Vehicle capacity.
            revenue: Revenue per unit.
            cost_unit: Cost per distance unit.
            values: Merged config values.
            mandatory_nodes: Local indices of nodes that MUST be visited.
            kwargs: Additional solver-specific arguments.

        Returns:
            Tuple of (routes, profit, solver_cost).
        """

        pass
