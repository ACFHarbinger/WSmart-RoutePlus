"""
Simulation state encapsulation and consolidation.

This module provides the SimulationDayContext dataclass and orchestrates
the execution of a single simulation day using the Command Pattern.

Attributes:
    SimulationDayContext: Represents the context of a single simulation day.

Example:
    >>> from logic.src.pipeline.simulations.day_context import SimulationDayContext
    >>> context = SimulationDayContext(graph_size=51)
    >>> context.graph_size
    51
"""

from __future__ import annotations

import os
import random
import re
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from multiprocessing.synchronize import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

if TYPE_CHECKING:
    from logic.src.pipeline.simulations.bins import Bins

import numpy as np
import pandas as pd
import torch

from logic.src.constants import DAY_METRICS


def get_canonical_policy_name(policy_name: str) -> str:
    """
    Get canonical algorithm name for RNG seeding.

    This extracts the base algorithm name from the policy string to ensure
    consistent seeding across different selection strategies.

    Args:
        policy_name: Full policy name (e.g., 'lookahead_ma_ts_custom_gamma3')

    Returns:
        Canonical name for seeding (e.g., 'ma_ts')

    Examples:
        >>> get_canonical_policy_name('lookahead_ma_ts_custom_gamma3')
        'ma_ts'
    """
    parts = policy_name.lower().split("_")

    # Try to find algorithm name (typically after lookahead/policy prefix)
    for _i, part in enumerate(parts):
        if part in ["lookahead", "policy", "regular", "lastminute", "revenue"]:
            # Skip selection strategy prefixes
            continue
        if part in ["custom", "gamma", "gamma1", "gamma2", "gamma3"]:
            # Skip config suffixes
            break
        # Found algorithm name
        return part

    # Fallback: return original name
    return policy_name


def _clean(name: Any) -> str:  # noqa: C901
    """Clean a policy/strategy name for display."""
    if name is None:
        return "None"

    # Handle lists/sequences (including OmegaConf ListConfig and nested lists)
    # Recursively unwrap until we find a string or a dict/config
    if not isinstance(name, str) and hasattr(name, "__iter__") and not hasattr(name, "items"):
        try:
            name_list = list(name)
            if len(name_list) > 0:
                # Try each item until we find something valid
                for item in name_list:
                    cleaned = _clean(item)
                    if cleaned != "None":
                        return cleaned
            return "None"
        except Exception:
            return "None"

    # Handle string names/paths
    if isinstance(name, str):
        if name.lower() in ["none", "null", "false", "[]", "none.yaml"]:
            return "None"

        # If it's a full path, get the basename
        base = os.path.basename(name)
        # Remove common prefixes and extensions
        for p in ["ms_", "ri_", "ac_", "policy_", ".yaml", ".xml", ".json"]:
            base = base.replace(p, "")

        # Clean up any remaining directory parts if basename failed (e.g. windows paths)
        if "/" in base:
            base = base.split("/")[-1]
        if "\\" in base:
            base = base.split("\\")[-1]

        return base.replace("_", " ").title()

    # Handle structured config objects (Dataclasses, Dicts, DictConfigs)
    if hasattr(name, "strategy"):
        return _clean(name.strategy)

    if hasattr(name, "get") or isinstance(name, Mapping):
        # Handle new dict format: {"other/ms_last_minute.yaml": "cf70"}
        if len(name) == 1:
            key, val = next(iter(name.items()))
            key_str = str(key)
            if key_str.endswith(".yaml") or key_str.endswith(".xml"):
                raw_v = val
                if hasattr(val, "__iter__") and not isinstance(val, str):
                    try:
                        v_list = list(val)
                        raw_v = v_list[0] if v_list else ""
                    except Exception:
                        pass
                variant = str(raw_v) if raw_v is not None else ""
                if variant and variant != "default":
                    return variant.replace("_", " ").title()
                base = os.path.basename(key_str)
                for p in ["ms_", "ri_", "ac_", "policy_", ".yaml", ".xml", ".json"]:
                    base = base.replace(p, "")
                return base.replace("_", " ").title()

        # Check exhaustive list of common keys for strategies
        for k in ["strategy", "type", "name", "id", "model", "method"]:
            val = name.get(k) if hasattr(name, "get") else name.get(k, None)
            if val:
                cleaned = _clean(val)
                if cleaned != "None":
                    return cleaned
        return "Custom"

    return "None"


# Recursive helper to find naming keys in potentially nested/listed configs
def find_policy_keys(obj: Any) -> Dict[str, Any]:
    """
    Recursively find mandatory_selection and route_improvement keys in a config.
    Searches for synonyms and handles deep nesting.

    Args:
        obj: The configuration object (dict, list, DictConfig, etc.)

    Returns:
        Dictionary containing the found keys.
    """
    found = {}

    # Synonyms for metadata keys
    MS_KEYS = {"mandatory_selection", "ms", "node_selection", "selection_strategy"}
    RI_KEYS = {"route_improvement", "ri", "improvement_strategy", "local_search"}
    AC_KEYS = {"acceptance_criteria", "acceptance_criterion", "ac"}

    # Handle Mapping (dict, DictConfig)
    if isinstance(obj, Mapping) or hasattr(obj, "items"):
        # 1. Check current level for direct matches or synonyms
        for key, value in obj.items():
            if key in MS_KEYS:
                if value is not None and str(value).lower() not in ["none", "null", "false", "[]"]:
                    found["mandatory_selection"] = value
            elif key in RI_KEYS and value is not None and str(value).lower() not in ["none", "null", "false", "[]"]:
                found["route_improvement"] = value
            elif key in AC_KEYS and value is not None and str(value).lower() not in ["none", "null", "false", "[]"]:
                found["acceptance_criteria"] = value

        # 2. Recurse into all values to find deeper configurations
        # We process values in order, so deeper values will overwrite shallower ones
        for v in obj.values():
            if v is not None and not isinstance(v, (str, int, float, bool)):
                inner = find_policy_keys(v)
                if inner:
                    found.update(inner)

    # Handle sequences (list, ListConfig)
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        for item in obj:
            if item is not None and not isinstance(item, (str, int, float, bool)):
                inner = find_policy_keys(item)
                if inner:
                    found.update(inner)

    return found


def build_naming_config(pol_cfg: Any, global_sim_cfg: Any) -> Dict[str, Any]:
    """
    Build a configuration dictionary specifically for naming resolution.
    It combines keys found in the policy config with global simulation defaults.

    Args:
        pol_cfg: The policy-specific configuration.
        global_sim_cfg: The global simulation configuration (ctx.cfg.sim).

    Returns:
        A dictionary with 'mandatory_selection' and 'route_improvement' keys.
    """
    # 1. Extract keys from policy config (recursive)
    extracted = find_policy_keys(pol_cfg)

    # 2. Fallback to global simulation defaults if keys are missing or None
    for key in ["mandatory_selection", "route_improvement", "acceptance_criteria"]:
        if not extracted.get(key) and hasattr(global_sim_cfg, key):
            val = getattr(global_sim_cfg, key)
            if val:
                extracted[key] = val

    return extracted


def resolve_policy_display_name(policy: Any, sim_cfg: Any) -> Tuple[str, str]:
    """Resolves the original policy ID and its full descriptive display name.

    Args:
        policy: The policy configuration (string or dict).
        sim_cfg: The simulation configuration.

    Returns:
        Tuple of (pol_id_orig, display_name)
    """
    from logic.src.utils.configs.config_loader import load_config
    from logic.src.utils.infrastructure.setup_sims import deep_sanitize, get_pol_name

    pol_id_orig = get_pol_name(policy)
    sanitized_policy = deep_sanitize(policy)
    pol_cfg = {}

    if isinstance(sanitized_policy, str):
        config_paths = deep_sanitize(sim_cfg.config_path) if hasattr(sim_cfg, "config_path") else {}
        if config_paths is None:
            config_paths = {}
        if pol_id_orig in config_paths:
            loaded_cfg = deep_sanitize(config_paths[pol_id_orig])
            if isinstance(loaded_cfg, dict):
                pol_cfg = loaded_cfg
            elif isinstance(loaded_cfg, str) and os.path.exists(loaded_cfg):
                try:
                    pol_cfg = load_config(loaded_cfg)
                except Exception:
                    pol_cfg = {}
    elif isinstance(sanitized_policy, dict) and len(sanitized_policy) == 1 and pol_id_orig in sanitized_policy:
        pol_cfg = sanitized_policy[pol_id_orig]
    elif isinstance(sanitized_policy, dict):
        pol_cfg = sanitized_policy

    sanitized_pol_cfg = deep_sanitize(pol_cfg)
    naming_cfg = build_naming_config(sanitized_pol_cfg, sim_cfg)
    display_name = get_full_policy_name(pol_id_orig, naming_cfg)

    return pol_id_orig, display_name


def to_slug(name: str) -> str:
    """Converts a display name to a safe slug string for IDs and filenames.

    Strips parenthetical distribution tags (e.g. " (Gamma3)", " (Emp)") so
    they appear in human-readable output but never reach the log filename.
    Replaces ' + ' with '_', removes remaining '()' chars, and lowercases.

    Args:
        name: The display name to convert.

    Returns:
        The slug string.
    """
    # Strip " (Gamma3)"-style parenthetical suffixes before building the slug
    name = re.sub(r"\s*\([^)]*\)", "", name)
    return name.replace(" + ", "_").replace(" ", "_").replace("(", "").replace(")", "").lower()


def get_full_policy_name(pol_name: str, config: Dict[str, Any]) -> str:
    """
    Construct a descriptive policy name including mandatory selection and route improvement.

    This reflects the entire construction and improvement stack.

    Args:
        pol_name: Base policy name (the route constructor).
        config: The naming configuration dictionary.

    Returns:
        A formatted name string (e.g., 'Lookahead + AM + None').
    """

    # 1. Extract Mandatory Selection
    ms = config.get("mandatory_selection")
    ms_name = _clean(ms)

    # 2. Extract Route Construction (Base Policy)
    # Strip common prefixes/suffixes
    base_id = pol_name.lower().replace("policy_", "")

    # Remove distribution suffix and anything after it
    for suffix in ["_emp", "_gamma"]:
        if suffix in base_id:
            base_id = base_id.split(suffix)[0]

    if "_" in base_id:
        # If it looks expanded (e.g., 'ms_regular_alns_ri_none'), try to extract the middle
        parts = base_id.split("_")
        if "ms" in parts or "ri" in parts:
            # It's already expanded, try to find the 'constructor' part
            # This is a bit heuristic but should work for most cases
            for p in parts:
                if p not in ["ms", "ri", "none", "custom", "alns", "hgs", "new", "og"]:
                    base_id = p
                    break

    # 3. Extract Route Improvement
    ri = config.get("route_improvement")
    ri_name = _clean(ri)

    # Remove redundant mandatory selection name from base_id to prevent duplication
    ms_slug = ms_name.lower().replace(" ", "_")
    if ms_slug != "none" and ms_slug in base_id:
        base_id = base_id.replace(f"_{ms_slug}", "").replace(f"{ms_slug}_", "").replace(ms_slug, "")

    # 4. Extract Acceptance Criteria
    ac = config.get("acceptance_criteria")
    ac_name = _clean(ac)

    # Remove redundant route improvement name from base_id to prevent duplication
    ri_slug = ri_name.lower()
    if ri_slug != "none" and ri_slug in base_id:
        base_id = base_id.replace(f"_{ri_slug}", "").replace(f"{ri_slug}_", "").replace(ri_slug, "")

    # Remove redundant acceptance criteria name from base_id to prevent duplication
    ac_slug = ac_name.lower()
    if ac_slug != "none" and ac_slug in base_id:
        base_id = base_id.replace(f"_{ac_slug}", "").replace(f"{ac_slug}_", "").replace(ac_slug, "")

    # Extract distribution suffix for display only (most specific first)
    dist_tag = ""
    for suffix in ["_gamma3", "_gamma2", "_gamma1", "_gamma", "_emp"]:
        if suffix in pol_name.lower():
            dist_tag = f" ({suffix.lstrip('_').capitalize()})"
            break

    base_name = base_id.upper()
    if ac_name.lower() != "none":
        return f"{ms_name} + {base_name} + {ac_name} + {ri_name}{dist_tag}"
    else:
        return f"{ms_name} + {base_name} + {ri_name}{dist_tag}"


@dataclass
class SimulationDayContext(Mapping):
    """
    Context object encapsulating the state of a simulation day.

    Attributes:
        graph_size: Total nodes in the problem graph, **including** the depot (= num_loc + 1).
        full_policy: Full string identifier of the policy (e.g., 'policy_regular3_gamma1').
        policy_name: Name of the policy.
        bins: The Bins object managing waste levels.
        new_data: DataFrame containing new data for the day.
        coords: DataFrame containing node coordinates.
        distance_matrix: Matrix of distances between nodes.
        distpath_tup: Tuple (dist_matrix, paths, dm_tensor, distC).
        distancesC: Integer distance matrix (numpy).
        paths_between_states: Precomputed paths.
        dm_tensor: Tensor version of distance matrix.
        sample_id: ID of the current sample.
        overflows: Current count of overflows.
        day: Current day index.
        model_env: The model environment object.
        model_ls: Tuple of model components.
        n_vehicles: Number of vehicles.
        area: Area identifier.
        realtime_log_path: Path for real-time logging.
        waste_type: Type of waste.
        current_collection_day: Index of the current collection day.
        cached: Cached data.
        device: Torch device.
        lock: Multiprocessing lock.
        hrl_manager: Manager for Hierarchical RL.
        config: Configuration dictionary.
        cost_weight: Weight for length.
        waste_weight: Weight for waste.
        overflow_penalty: Weight for overflows.
        engine: The policy engine type (e.g., 'neural', 'classical').
        threshold: Decision threshold for bin selection.
        seed: Base seed for the simulation.
        policy_seed: Policy-specific seed for RNG isolation.
        display_name: Descriptive name for logging (e.g., 'Regular + ALNS + None').

        # Mutable attributes added during run_day
        daily_log: Dictionary for daily logs.
        output_dict: Dictionary for output results.
        tour: The generated tour.
        cost: The cost of the route.
        profit: The profit from collection.
        collected: IDs of collected bins.
        total_collected: Total amount collected.
        ncol: Number of collections.
        new_overflows: Number of new overflows today.
        sum_lost: Amount of waste lost.
        fill: Fill levels.
        total_fill: Total fill levels.
        extra_output: Any extra output from policy.
        mandatory: List of mandatory bins.
        time: Elapsed time for policy execution.
    """

    # Required/Core Fields
    graph_size: int = 0
    full_policy: str = ""
    policy_name: str = ""
    bins: Optional[Bins] = None
    new_data: Optional[pd.DataFrame] = None
    coords: Optional[pd.DataFrame] = None
    distance_matrix: Optional[Union[np.ndarray, List[List[float]]]] = None
    distpath_tup: Tuple[Any, ...] = (None, None, None, None)
    distancesC: Optional[np.ndarray] = None
    paths_between_states: Optional[Dict[Tuple[int, int], List[int]]] = None
    dm_tensor: Optional[torch.Tensor] = None
    sample_id: int = 0
    overflows: int = 0
    day: int = 0
    model_env: Any = None
    model_ls: Tuple[Any, ...] = (None,)
    n_vehicles: int = 1
    area: str = ""
    realtime_log_path: Optional[str] = None
    waste_type: str = ""
    current_collection_day: int = 0
    cached: Optional[List[int]] = None
    device: Optional[torch.device] = None
    lock: Optional[Lock] = None
    hrl_manager: Any = None
    config: Optional[Dict[str, Any]] = None
    cost_weight: float = 1.0
    waste_weight: float = 1.0
    overflow_penalty: float = 1.0
    engine: Optional[str] = None
    threshold: Optional[float] = None
    seed: int = 42
    policy_seed: Optional[int] = None  # Policy-specific seed for RNG isolation
    display_name: str = ""
    problem: str = "vrpp"
    shift_hours: float = 7.0
    time_matrix: Optional[np.ndarray] = None
    avg_speed_kmh: float = 35.0
    service_time_h: float = 1.5 / 60.0
    vehicle_capacity: float = 100.0

    # Optional/Mutable Fields
    daily_log: Optional[Dict[str, Any]] = None
    output_dict: Optional[Dict[str, Any]] = None
    tour: Optional[List[int]] = None
    cost: float = 0.0
    profit: float = 0.0
    collected: Optional[np.ndarray] = None
    total_collected: float = 0.0
    ncol: int = 0
    new_overflows: int = 0
    sum_lost: float = 0.0
    fill: Optional[np.ndarray] = None
    total_fill: Optional[np.ndarray] = None
    extra_output: Any = None
    mandatory: Optional[List[int]] = None
    time: float = 0.0
    time_spent: float = 0.0

    @property
    def field_names(self):
        """Returns the names of all fields in the dataclass.

        Returns:
            List of field names.
        """
        return [f.name for f in fields(self)]

    def __post_init__(self):
        """Initialize any derived or default state if needed."""
        if self.config is None:
            self.config = {}

    def __getitem__(self, key: str) -> Any:
        """Dictionary-like access to context fields.

        Args:
            key: The key to access.

        Returns:
            The value associated with the key.
        """
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow setting context fields via dictionary syntax.

        Args:
            key: The name of the field to set.
            value: The value to assign to the field.
        """
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Safely get a value from the context with an optional default.

        Args:
            key: The name of the field to retrieve.
            default: The default value to return if the field does not exist.

        Returns:
            The value of the specified field or the default.
        """
        return getattr(self, key, default)

    def __iter__(self):
        """Return an iterator over field names to support Mapping interface.

        Returns:
            Iterator over the field names.
        """
        return iter((f.name for f in fields(self)))

    def __len__(self):
        """Return the number of fields to support Mapping interface.

        Returns:
            Number of fields in the dataclass.
        """
        return len(fields(self))

    def __contains__(self, key: object) -> bool:
        """Check if a field exists in the context.

        Args:
            key: The name of the field to check.

        Returns:
            True if the field exists, False otherwise.
        """
        if not isinstance(key, str):
            return False
        return hasattr(self, key)


def set_daily_waste(
    model_data: Dict[str, Any],
    waste: np.ndarray,
    device: torch.device,
    fill: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Updates neural model input with current bin waste levels.

    Args:
        model_data: Dictionary of model input tensors.
        waste: Array of current waste levels for all bins.
        device: Target torch device.
        fill: Array of current fill levels for all bins.

    Returns:
        Updated model_data dictionary moved to the target device.
    """
    waste_tensor = torch.as_tensor(waste, dtype=torch.float32).unsqueeze(0).div(100.0)
    if device.type == "cuda":
        waste_tensor = waste_tensor.pin_memory()
    model_data["waste"] = waste_tensor

    if "fill_history" in model_data and fill is not None:
        fill_tensor = torch.as_tensor(fill, dtype=torch.float32).unsqueeze(0).div(100.0)
        if device.type == "cuda":
            fill_tensor = fill_tensor.pin_memory()
        model_data["current_fill"] = fill_tensor

    from logic.src.utils.functions import move_to

    return move_to(model_data, device, non_blocking=True)


def get_daily_results(
    total_collected: float,
    ncol: int,
    cost: float,
    tour: List[int],
    day: int,
    new_overflows: int,
    sum_lost: float,
    coordinates: pd.DataFrame,
    profit: float,
    time: float,
    mandatory_nodes: Optional[List[int]] = None,
    time_spent: Optional[float] = None,
) -> Dict[str, Union[int, float, List[Union[int, str]]]]:
    """Formats raw simulation outputs into structured daily log dictionary.

    Args:
        total_collected: Total weight of waste collected (kg).
        ncol: Number of bins collected.
        cost: Total distance traveled (km).
        tour: List of node indices in the collection route.
        day: Current simulation day index.
        new_overflows: Number of new overflows occurred.
        sum_lost: Amount of waste lost due to overflows (kg).
        coordinates: DataFrame containing bin metadata and coordinates.
        profit: Total profit from collected waste.
        time: Execution time of the routing policy (s).
        mandatory_nodes: Optional list of bin indices selected as mandatory
            before routing (iloc-based). Resolved to real IDs.
        time_spent: Optional total shift time spent on route and services (h).

    Returns:
        Dictionary containing formatted daily metrics and the route.
    """
    dlog: Dict[str, Any] = {key: 0 for key in DAY_METRICS}
    dlog["day"] = day
    dlog["overflows"] = new_overflows
    dlog["kg_lost"] = sum_lost
    dlog["time"] = time
    if time_spent is not None:
        dlog["time_spent"] = float(time_spent)
    mandatory_ids: List[int] = []
    for idx in mandatory_nodes or []:
        try:
            mandatory_ids.append(int(coordinates.iloc[idx]["ID"]))
        except (IndexError, KeyError, TypeError, ValueError):
            mandatory_ids.append(idx)
    dlog["mandatory_nodes"] = mandatory_ids

    if tour and len(tour) > 2:
        reward = total_collected - new_overflows - cost
        dlog["kg"] = total_collected
        dlog["ncol"] = ncol
        dlog["km"] = cost
        dlog["kg/km"] = total_collected / cost if cost > 0 else 0
        dlog["reward"] = reward
        dlog["profit"] = profit
        coordinate_ids = coordinates["ID"].tolist()
        # Preserve trip boundaries so saved tours reproduce capacity/time checks.
        dlog["tour"] = [0 if node == 0 else coordinate_ids[node] for node in tour]
    else:
        dlog["kg"] = 0
        dlog["ncol"] = 0
        dlog["km"] = 0
        dlog["kg/km"] = 0
        dlog["reward"] = -new_overflows
        dlog["profit"] = 0
        dlog["tour"] = [0]
    return dlog


def run_day(context: SimulationDayContext) -> SimulationDayContext:
    """
    Orchestrates a single simulation day using the Command Pattern.
    Args:
        context: The simulation context for the day.

    Returns:
        The updated context after executing all daily actions.
    """
    # Compute policy-specific seed for RNG isolation
    canonical_name = get_canonical_policy_name(context.policy_name)
    name_hash = zlib.adler32(canonical_name.encode()) & 0xFFFFFFFF
    policy_seed = (context.seed + name_hash + context.day) % (2**31)
    context.policy_seed = policy_seed  # Store for later use

    # Set policy-specific RNG state (isolates this policy from others)
    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)

    # CRITICAL: Reset Bins RNG to policy-specific state
    # This ensures bin fill predictions are identical for equivalent algorithms
    # regardless of execution order
    if context.bins is not None:
        # Reset the Bins RNG using policy-specific seed + day + sample_id
        # This gives each (policy, day, sample) tuple its own isolated bin predictions
        bins_seed = (policy_seed + context.day + context.sample_id) % (2**31)
        context.bins.rng = np.random.default_rng(bins_seed)

    from logic.src.pipeline.simulations.actions import (
        CollectAction,
        FillAction,
        LogAction,
        MandatorySelectionAction,
        RouteConstructionAction,
        RouteImprovementAction,
    )
    from logic.src.pipeline.simulations.actions.time_constraints import TimeConstraintAction

    commands = [
        FillAction(),
        MandatorySelectionAction(),
        RouteConstructionAction(),
        RouteImprovementAction(),
        TimeConstraintAction(),
        CollectAction(),
        LogAction(),
    ]

    for command in commands:
        command.execute(cast(Dict[str, Any], context))

    return context
