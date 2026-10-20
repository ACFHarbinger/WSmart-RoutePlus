"""

Template generators live in ``search_space_templates``; this module keeps the runtime search-space loaders and re-exports those names.
Search space definitions for simulation policies.

This module contains the default hyperparameter search spaces for all supported
simulation policies, including ALNS, HGS, ACO, and SISR. Each policy's search
space is defined as a dictionary with parameter names as keys and their HPO
specifications as values.

Search space JSON files are stored in sub-directories next to this module:
  - routing/             Routing algorithm parameter search spaces.
  - route_improvement/   Route improver parameter search spaces.
  - packing/             Packing algorithm parameter search spaces.
  - acceptance_criteria/ Move acceptance criteria parameter search spaces.

Attributes:
    ROUTING_SPACES: Dictionary mapping routing algorithm names to their search spaces,
        loaded eagerly from the routing/ directory at import time.

Functions:
    get_search_space:           Retrieve the search space for a specific policy.
    validate_search_space:      Validate a search space before running trials.
    load_all_search_spaces:     Load all JSON search spaces from the routing/ directory.
    generate_routing_templates:             Scaffold JSON templates from policy dataclass configs.
    generate_route_improvement_templates:   Scaffold JSON templates for route improvers.
    generate_packing_templates:             Scaffold JSON templates for packing algorithms.
    generate_acceptance_criteria_templates: Scaffold JSON templates for acceptance criteria.

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

import json
import os
from typing import Any, Dict, List, Optional, Union

from .search_space_templates import (
    _extract_params_from_config as _extract_params_from_config,
)
from .search_space_templates import (
    generate_acceptance_criteria_templates as generate_acceptance_criteria_templates,
)
from .search_space_templates import (
    generate_packing_templates as generate_packing_templates,
)
from .search_space_templates import (
    generate_route_improvement_templates as generate_route_improvement_templates,
)
from .search_space_templates import (
    generate_routing_templates as generate_routing_templates,
)

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------

# Packing algorithm parameter search spaces
PACKING_DIR = os.path.join(os.path.dirname(__file__), "packing")

# Route improver parameter search spaces
ROUTE_IMPROVEMENT_DIR = os.path.join(os.path.dirname(__file__), "route_improvement")

# Routing algorithm parameter search spaces  (primary source loaded at import)
ROUTING_DIR = os.path.join(os.path.dirname(__file__), "routing")

# Move acceptance criteria parameter search spaces
ACCEPTANCE_CRITERIA_DIR = os.path.join(os.path.dirname(__file__), "acceptance_criteria")

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
        space: Mapping of parameter names to spec dicts.
        policy_name: Used only for contextual error messages.

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
                    f"key(s) {missing}. Edit hpo/routing/{policy_name}.json."
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
        directory: Path to the directory containing JSON files.
            Defaults to :data:`ROUTING_DIR` if not supplied.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: Mapping of lowercase policy names
            to their search space specification dicts.
    """
    target_dir = directory or ROUTING_DIR
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
PACKING_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = load_all_search_spaces(PACKING_DIR)
ROUTE_IMPROVEMENT_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = load_all_search_spaces(ROUTE_IMPROVEMENT_DIR)
ROUTING_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = load_all_search_spaces(ROUTING_DIR)
ACCEPTANCE_CRITERIA_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = load_all_search_spaces(ACCEPTANCE_CRITERIA_DIR)


def get_component_search_space(
    component_type: str,
    name: str,
    raise_on_invalid: bool = True,
    keywords: Optional[str] = None,
    index: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Retrieve search space for a specific component with its config path prefix.

    Component types and their associated prefixes:
      - 'packing'             -> 'packing.0.'
      - 'route_improvement'   -> 'route_improvement.0.'
      - 'acceptance_criteria' -> 'acceptance_criterion.params.'
      - 'routing'             -> '' (relative to policy config root)

    Args:
        component_type: One of 'packing', 'route_improvement', 'acceptance_criteria', 'routing'.
        name: Name of the component (e.g., 'boltzmann', 'alns').
        raise_on_invalid: Whether to validate the space before returning.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of prefixed config paths to HPO specs.
    """
    registry_map = {
        "packing": (PACKING_SPACES, "packing.{idx}."),
        "route_improvement": (ROUTE_IMPROVEMENT_SPACES, "route_improvement.{idx}."),
        "acceptance_criteria": (ACCEPTANCE_CRITERIA_SPACES, "acceptance_criterion.params."),
        "routing": (ROUTING_SPACES, ""),
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
    routing: Optional[str] = None,
    packing: Optional[Union[str, List[str]]] = None,
    route_improvement: Optional[Union[str, List[str]]] = None,
    acceptance_criteria: Optional[Union[str, List[str]]] = None,
    routing_keywords: Optional[str] = None,
    packing_keywords: Optional[str] = None,
    route_improvement_keywords: Optional[str] = None,
    acceptance_criteria_keywords: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compose a full search space from multiple components.

    Args:
        routing: Name of the routing algorithm (from routing/).
        packing: Name(s) of the packing algorithm(s) (from packing/).
        route_improvement: Name(s) of the improver(s) (from route_improvement/).
        acceptance_criteria: Name(s) of the acceptance criterion (from acceptance_criteria/).
        routing_keywords: Keep only routing parameters matching these keywords.
        packing_keywords: Keep only packing parameters matching these keywords.
        route_improvement_keywords: Keep only improver parameters matching these keywords.
        acceptance_criteria_keywords: Keep only acceptance parameters matching these keywords.

    Returns:
        Dict[str, Dict[str, Any]]: Combined search space with appropriate prefixes.
    """
    composed: Dict[str, Dict[str, Any]] = {}

    if routing:
        composed.update(get_component_search_space("routing", routing, keywords=routing_keywords))

    # Handle multiple packing algorithms
    if packing:
        packings = [packing] if isinstance(packing, str) else packing
        for idx, p_name in enumerate(packings):
            composed.update(get_component_search_space("packing", p_name, keywords=packing_keywords, index=idx))

    # Handle multiple route improvers
    if route_improvement:
        improvers = [route_improvement] if isinstance(route_improvement, str) else route_improvement
        for idx, i_name in enumerate(improvers):
            composed.update(
                get_component_search_space("route_improvement", i_name, keywords=route_improvement_keywords, index=idx)
            )

    # Acceptance criteria - usually single, but keep consistent
    if acceptance_criteria:
        criteria = [acceptance_criteria] if isinstance(acceptance_criteria, str) else acceptance_criteria
        for c_name in criteria:
            composed.update(
                get_component_search_space("acceptance_criteria", c_name, keywords=acceptance_criteria_keywords)
            )

    return composed


def get_search_space(policy_name: str, raise_on_invalid: bool = True) -> Dict[str, Dict[str, Any]]:
    """Retrieve and validate the search space for a specific policy.

    Args:
        policy_name: Name of the policy (e.g., 'alns', 'hgs').
        raise_on_invalid: When True (default), raises a ValueError if the
            search space contains invalid parameter specifications.  Set to False
            to allow partial/template spaces during development.

    Returns:
        Dict[str, Dict[str, Any]]: The validated search space specification
            dictionary, or an empty dict if no space is registered for the policy.

    Raises:
        ValueError: If ``raise_on_invalid=True`` and the space has invalid specs.
    """
    space = ROUTING_SPACES.get(policy_name.lower(), {})

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



# ---------------------------------------------------------------------------
# Scaffold generators (called from __main__ or a management command)
# ---------------------------------------------------------------------------






if __name__ == "__main__":
    generate_routing_templates()
    generate_route_improvement_templates()
    generate_packing_templates()
    generate_acceptance_criteria_templates()
