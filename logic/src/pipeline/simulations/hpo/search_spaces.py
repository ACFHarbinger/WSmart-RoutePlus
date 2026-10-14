"""
Search space definitions for simulation policies.

This module contains the default hyperparameter search spaces for all supported
simulation policies, including ALNS, HGS, ACO, and SISR. Each policy's search
space is defined as a dictionary with parameter names as keys and their HPO
specifications as values.

Search space JSON files are stored in sub-directories next to this module:
  - jobs/         Routing algorithm parameter search spaces.
  - interceptors/ Route improver parameter search spaces.
  - filters/      Mandatory node selection parameter search spaces.
  - rules/        Move acceptance criteria parameter search spaces.

Attributes:
    POLICY_SEARCH_SPACES: Dictionary mapping policy names to their search spaces,
        loaded eagerly from the jobs/ directory at import time.

Functions:
    get_search_space:           Retrieve the search space for a specific policy.
    validate_search_space:      Validate a search space before running trials.
    load_all_search_spaces:     Load all JSON search spaces from the jobs/ directory.
    generate_policy_filters:            Scaffold JSON templates from policy dataclass configs.
    generate_route_improvement_interceptors: Scaffold JSON templates for route improvers.
    generate_packing_jobs:  Scaffold JSON templates for mandatory selectors.
    generate_acceptance_criteria_rules: Scaffold JSON templates for acceptance criteria.

Example:
    >>> from logic.src.policies.helpers.hpo import get_search_space
    >>> alns_space = get_search_space("alns")
    >>> print(alns_space)
    {
        'max_iterations': {'type': 'int', 'low': 500, 'high': 10000, 'step': 500},
        'start_temp': {'type': 'float', 'low': 1.0, 'high': 500.0, 'log': True},
        ...
    }
"""

import dataclasses
import importlib
import inspect
import json
import os
import pkgutil
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

# Packing algorithm parameter search spaces
FILTERS_DIR = os.path.join(os.path.dirname(__file__), "filters")

# Route improver parameter search spaces
INTERCEPTORS_DIR = os.path.join(os.path.dirname(__file__), "interceptors")

# Routing algorithm parameter search spaces  (primary source loaded at import)
JOBS_DIR = os.path.join(os.path.dirname(__file__), "jobs")

# Move acceptance criteria parameter search spaces
RULES_DIR = os.path.join(os.path.dirname(__file__), "rules")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_VALID_TYPES = {"float", "int", "categorical"}


def validate_search_space(space: Dict[str, Any], policy_name: str) -> List[str]:
    """Validate a search space specification and return a list of error messages.

    Validation is intentionally non-raising here so callers can decide whether
    to emit warnings (at load time) or raise hard errors (before trial start).
    Use ``raise_on_invalid=True`` in :func:`get_search_space` for the strict mode.

    Rules checked:
      - Each parameter must have a known ``type`` ('float', 'int', 'categorical').
      - float/int parameters must have both ``low`` and ``high``, with low < high.
      - categorical parameters must have a non-empty ``choices`` list.

    Args:
        space (Dict[str, Any]): Mapping of parameter names to spec dicts.
        policy_name (str): Used only for contextual error messages.

    Returns:
        List[str]: List of human-readable error strings; empty if space is valid.
    """
    errors: List[str] = []
    for name, spec in space.items():
        # Skip comments or metadata keys starting with an underscore
        if name.startswith("_"):
            continue

        if not isinstance(spec, dict):
            errors.append(f"[{policy_name}] '{name}': spec must be a dictionary, got {type(spec).__name__}.")
            continue

        p_type = spec.get("type")
        if p_type not in _VALID_TYPES:
            errors.append(f"[{policy_name}] '{name}': unknown type '{p_type}'. Must be one of {sorted(_VALID_TYPES)}.")
            continue  # Cannot validate further without a known type.

        if p_type in ("float", "int"):
            missing = [k for k in ("low", "high") if k not in spec]
            if missing:
                errors.append(
                    f"[{policy_name}] '{name}' (type='{p_type}'): missing required "
                    f"key(s) {missing}. Edit hpo/jobs/{policy_name}.json."
                )
            elif spec["low"] >= spec["high"]:
                errors.append(
                    f"[{policy_name}] '{name}': 'low' ({spec['low']}) must be "
                    f"strictly less than 'high' ({spec['high']})."
                )

        elif p_type == "categorical":
            if not spec.get("choices"):
                errors.append(f"[{policy_name}] '{name}' (type='categorical'): 'choices' is empty or missing.")

    return errors


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_all_search_spaces(directory: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load all search space definitions from a directory of JSON files.

    Each JSON file is named ``<policy_name>.json`` and maps parameter names to
    their HPO specification dicts.  Files that fail to parse are skipped with a
    printed warning; validation errors are printed as warnings but do not prevent
    loading so that partially-configured files can still be used during development.

    Args:
        directory (Optional[str]): Path to the directory containing JSON files.
            Defaults to :data:`JOBS_DIR` if not supplied.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: Mapping of lowercase policy names
            to their search space specification dicts.
    """
    target_dir = directory or JOBS_DIR
    spaces: Dict[str, Dict[str, Dict[str, Any]]] = {}

    if not os.path.exists(target_dir):
        return spaces

    for filename in sorted(os.listdir(target_dir)):
        if not filename.endswith(".json"):
            continue

        policy_name = filename[:-5].lower()
        filepath = os.path.join(target_dir, filename)

        try:
            with open(filepath, "r") as fh:
                space = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[hpo/search_spaces] Could not load '{filepath}': {exc}")
            continue

        # Soft-validate: warn about incomplete specs but still register the space
        # so that developers can iterate on partial search spaces.
        errors = validate_search_space(space, policy_name)
        for err in errors:
            print(f"[hpo/search_spaces] WARNING – {err}")

        spaces[policy_name] = space

    return spaces


# Eagerly loaded at import time from the respective directories.
FILTER_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = load_all_search_spaces(FILTERS_DIR)
INTERCEPTOR_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = load_all_search_spaces(INTERCEPTORS_DIR)
JOB_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = load_all_search_spaces(JOBS_DIR)
RULE_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = load_all_search_spaces(RULES_DIR)

# Maintain backward compatibility for existing callers.
POLICY_SEARCH_SPACES = JOB_SPACES


def get_component_search_space(
    component_type: str,
    name: str,
    raise_on_invalid: bool = True,
    keywords: Optional[str] = None,
    index: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Retrieve search space for a specific component with its config path prefix.

    Component types and their associated prefixes:
      - 'filter'      -> 'packing.0.'
      - 'interceptor' -> 'route_improvement.0.'
      - 'rule'        -> 'acceptance_criterion.params.'
      - 'job'         -> '' (relative to policy config root)

    Args:
        component_type (str): One of 'filter', 'interceptor', 'rule', 'job'.
        name (str): Name of the component (e.g., 'boltzmann', 'alns').
        raise_on_invalid (bool): Whether to validate the space before returning.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of prefixed config paths to HPO specs.
    """
    registry_map = {
        "filter": (FILTER_SPACES, "packing.{idx}."),
        "interceptor": (INTERCEPTOR_SPACES, "route_improvement.{idx}."),
        "rule": (RULE_SPACES, "acceptance_criterion.params."),
        "job": (JOB_SPACES, ""),
    }

    if component_type not in registry_map:
        raise ValueError(f"Unknown component type: {component_type}. Use one of {list(registry_map.keys())}")

    registry, prefix_template = registry_map[component_type]

    # Apply index if provided, otherwise use .0. as default for compatibility
    if index is not None and "{idx}" in prefix_template:
        prefix = prefix_template.format(idx=index)
    elif "{idx}" in prefix_template:
        prefix = prefix_template.format(idx=0)
    else:
        prefix = prefix_template

    space = registry.get(name.lower(), {})

    if not space:
        return {}

    # Apply keyword filtering if requested.
    if keywords:
        kw_list = [k.strip().lower() for k in keywords.split(",") if k.strip()]
        if kw_list:
            space = {k: v for k, v in space.items() if any(kw in k.lower() for kw in kw_list)}

    if not space:
        return {}

    if raise_on_invalid:
        errors = validate_search_space(space, f"{component_type}:{name}")
        if errors:
            formatted = "\n  ".join(errors)
            raise ValueError(f"Search space for {component_type} '{name}' has errors:\n  {formatted}")

    return {f"{prefix}{k}": v for k, v in space.items()}


def compose_search_space(
    job: Optional[str] = None,
    filter: Optional[Union[str, List[str]]] = None,
    interceptor: Optional[Union[str, List[str]]] = None,
    rule: Optional[Union[str, List[str]]] = None,
    job_keywords: Optional[str] = None,
    filter_keywords: Optional[str] = None,
    interceptor_keywords: Optional[str] = None,
    rule_keywords: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compose a full search space from multiple components.

    Args:
        job (str, optional): Name of the main policy (from jobs/).
        filter (str, optional): Name of the selection strategy (from filters/).
        interceptor (str, optional): Name of the improver (from interceptors/).
        rule (str, optional): Name of the acceptance rule (from rules/).

    Returns:
        Dict[str, Dict[str, Any]]: Combined search space with appropriate prefixes.
    """
    composed: Dict[str, Dict[str, Any]] = {}

    if job:
        composed.update(get_component_search_space("job", job, keywords=job_keywords))

    # Handle multiple filters (mandatory selectors)
    if filter:
        filters = [filter] if isinstance(filter, str) else filter
        for idx, f_name in enumerate(filters):
            composed.update(get_component_search_space("filter", f_name, keywords=filter_keywords, index=idx))

    # Handle multiple interceptors (route improvers)
    if interceptor:
        interceptors = [interceptor] if isinstance(interceptor, str) else interceptor
        for idx, i_name in enumerate(interceptors):
            composed.update(get_component_search_space("interceptor", i_name, keywords=interceptor_keywords, index=idx))

    # Acceptance criteria (rule) - usually single, but keep consistent
    if rule:
        rules = [rule] if isinstance(rule, str) else rule
        for r_name in rules:
            composed.update(get_component_search_space("rule", r_name, keywords=rule_keywords))

    return composed


def get_search_space(policy_name: str, raise_on_invalid: bool = True) -> Dict[str, Dict[str, Any]]:
    """Retrieve and validate the search space for a specific policy.

    Args:
        policy_name (str): Name of the policy (e.g., 'alns', 'hgs').
        raise_on_invalid (bool): When True (default), raises a ValueError if the
            search space contains invalid parameter specifications.  Set to False
            to allow partial/template spaces during development.

    Returns:
        Dict[str, Dict[str, Any]]: The validated search space specification
            dictionary, or an empty dict if no space is registered for the policy.

    Raises:
        ValueError: If ``raise_on_invalid=True`` and the space has invalid specs.
    """
    space = POLICY_SEARCH_SPACES.get(policy_name.lower(), {})

    if space and raise_on_invalid:
        errors = validate_search_space(space, policy_name)
        if errors:
            formatted = "\n  ".join(errors)
            raise ValueError(
                f"Search space for policy '{policy_name}' has {len(errors)} error(s):\n"
                f"  {formatted}\n"
                f"Fix the corresponding JSON file before running HPO."
            )

    return space


# ---------------------------------------------------------------------------
# Config introspection helpers
# ---------------------------------------------------------------------------


def _extract_params_from_config(config_class: Any) -> Dict[str, Any]:
    """Extract parameters and their basic types from a dataclass config.

    Produces a *template* search space — numeric parameters will have ``type``
    set but will intentionally **not** include ``low``/``high`` bounds, since
    reasonable bounds are domain-specific and must be filled in by the user.
    A ``_NEEDS_BOUNDS`` sentinel is inserted so that :func:`validate_search_space`
    catches un-edited templates.

    Args:
        config_class (Any): A dataclass (not an instance) to introspect.

    Returns:
        Dict[str, Any]: Mapping of field names to template HPO spec dicts.
    """
    params: Dict[str, Any] = {}

    for field in dataclasses.fields(config_class):
        field_type = str(field.type).lower()

        if "bool" in field_type:
            param_spec: Dict[str, Any] = {"type": "categorical", "choices": [True, False]}
        elif "int" in field_type:
            # Intentionally omit low/high so validate_search_space flags the template.
            param_spec = {"type": "int", "_NEEDS_BOUNDS": True}
        elif "float" in field_type:
            param_spec = {"type": "float", "_NEEDS_BOUNDS": True}
        elif "str" in field_type:
            default_choice = field.default if field.default is not dataclasses.MISSING else None
            choices = [default_choice] if default_choice is not None else []
            param_spec = {"type": "categorical", "choices": choices}
        else:
            # Unsupported type — mark as object for manual handling.
            param_spec = {"type": "object"}

        # Attach default value when serialisable (informational, not used by suggest_param).
        if field.default is not dataclasses.MISSING:
            try:
                json.dumps(field.default)
                param_spec["default"] = field.default
            except (TypeError, OverflowError):
                param_spec["default"] = str(field.default)
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            param_spec["default"] = "factory"

        params[field.name] = param_spec

    return params


# ---------------------------------------------------------------------------
# Scaffold generators (called from __main__ or a management command)
# ---------------------------------------------------------------------------


def generate_policy_filters() -> None:
    """Scaffold JSON search space templates from policy dataclass configs.

    Scans the ``logic.src.configs.policies`` package, finds dataclass
    ``*Config`` classes, and writes template JSON files to :data:`JOBS_DIR`.
    Existing files are **not** overwritten to protect manually-tuned bounds.
    """
    from logic.src.configs import policies as policies_pkg

    os.makedirs(JOBS_DIR, exist_ok=True)
    package_path = policies_pkg.__path__

    for _, module_name, is_pkg in pkgutil.walk_packages(package_path):
        if is_pkg or module_name in ("abc", "__init__"):
            continue

        full_module_name = f"logic.src.configs.policies.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
        except Exception as exc:
            print(f"[generate_policy_filters] Error importing {full_module_name}: {exc}")
            continue

        config_class = None
        for name, obj in inspect.getmembers(module):
            if (
                name.endswith("Config")
                and dataclasses.is_dataclass(obj)
                and getattr(obj, "__module__", "") == full_module_name
            ):
                config_class = obj
                break

        if config_class is None:
            continue

        base_name = module_name.split(".")[-1].lower().replace("postconfig", "").replace("config", "")
        output_path = os.path.join(JOBS_DIR, f"{base_name}.json")

        if os.path.exists(output_path):
            print(f"[generate_policy_filters] Skipping existing file: {output_path}")
            continue

        params = _extract_params_from_config(config_class)
        with open(output_path, "w") as fh:
            json.dump(params, fh, indent=4)
        print(f"[generate_policy_filters] Generated: {output_path}")


def generate_route_improvement_interceptors() -> None:
    """Scaffold JSON search space templates for route improvement configs.

    Scans ``logic.src.configs.policies.other.route_improvement`` and writes
    template JSON files to :data:`INTERCEPTORS_DIR`.  Existing files are not
    overwritten.
    """
    from logic.src.configs.policies.other import route_improvement

    os.makedirs(INTERCEPTORS_DIR, exist_ok=True)

    for name, obj in inspect.getmembers(route_improvement):
        if not (
            (name.endswith("Config") or name.endswith("PostConfig"))
            and dataclasses.is_dataclass(obj)
            and getattr(obj, "__module__", "") == route_improvement.__name__
        ):
            continue

        base_name = name.lower().replace("postconfig", "").replace("config", "")
        output_path = os.path.join(INTERCEPTORS_DIR, f"{base_name}.json")

        if os.path.exists(output_path):
            print(f"[generate_route_improvement_interceptors] Skipping existing: {output_path}")
            continue

        params = _extract_params_from_config(obj)
        with open(output_path, "w") as fh:
            json.dump(params, fh, indent=4)
        print(f"[generate_route_improvement_interceptors] Generated: {output_path}")


def generate_packing_jobs() -> None:
    """Scaffold JSON search space templates for packing configs.

    Scans ``logic.src.configs.policies.other.packing`` and writes
    template JSON files to :data:`FILTERS_DIR`.  Existing files are not
    overwritten.
    """
    from logic.src.configs.policies.other import packing

    os.makedirs(FILTERS_DIR, exist_ok=True)

    for name, obj in inspect.getmembers(packing):
        if not (
            name.endswith("Config")
            and dataclasses.is_dataclass(obj)
            and getattr(obj, "__module__", "") == packing.__name__
        ):
            continue

        base_name = name.lower().replace("selectionconfig", "").replace("postconfig", "").replace("config", "")
        output_path = os.path.join(FILTERS_DIR, f"{base_name}.json")

        if os.path.exists(output_path):
            print(f"[generate_packing_jobs] Skipping existing: {output_path}")
            continue

        params = _extract_params_from_config(obj)
        with open(output_path, "w") as fh:
            json.dump(params, fh, indent=4)
        print(f"[generate_packing_jobs] Generated: {output_path}")


def generate_acceptance_criteria_rules() -> None:
    """Scaffold JSON search space templates for acceptance criteria configs.

    Scans ``logic.src.configs.policies.other.acceptance_criteria`` and writes
    template JSON files to :data:`RULES_DIR`. Existing files are not
    overwritten.
    """
    from logic.src.configs.policies.other import acceptance_criteria

    os.makedirs(RULES_DIR, exist_ok=True)

    for name, obj in inspect.getmembers(acceptance_criteria):
        if not (
            name.endswith("Config")
            and dataclasses.is_dataclass(obj)
            and getattr(obj, "__module__", "") == acceptance_criteria.__name__
        ):
            continue

        base_name = (
            name.lower()
            .replace("acceptanceconfig", "")
            .replace("selectionconfig", "")
            .replace("postconfig", "")
            .replace("config", "")
        )
        output_path = os.path.join(RULES_DIR, f"{base_name}.json")

        if os.path.exists(output_path):
            print(f"[generate_acceptance_criteria_rules] Skipping existing: {output_path}")
            continue

        params = _extract_params_from_config(obj)
        with open(output_path, "w") as fh:
            json.dump(params, fh, indent=4)
        print(f"[generate_acceptance_criteria_rules] Generated: {output_path}")


if __name__ == "__main__":
    generate_policy_filters()
    generate_route_improvement_interceptors()
    generate_packing_jobs()
    generate_acceptance_criteria_rules()
