"""Search-space JSON template generators.

Moved from ``logic.src.pipeline.simulations.hpo.search_spaces``.
"""

import dataclasses
import importlib
import inspect
import json
import os
import pkgutil
from typing import Any, Dict

PACKING_DIR = os.path.join(os.path.dirname(__file__), "packing")
ROUTE_IMPROVEMENT_DIR = os.path.join(os.path.dirname(__file__), "route_improvement")
ROUTING_DIR = os.path.join(os.path.dirname(__file__), "routing")
ACCEPTANCE_CRITERIA_DIR = os.path.join(os.path.dirname(__file__), "acceptance_criteria")


def _extract_params_from_config(config_class: Any) -> Dict[str, Any]:
    """Extract parameters and their basic types from a dataclass config.

    Produces a *template* search space — numeric parameters will have ``type``
    set but will intentionally **not** include ``low``/``high`` bounds, since
    reasonable bounds are domain-specific and must be filled in by the user.
    A ``_NEEDS_BOUNDS`` sentinel is inserted so that :func:`validate_search_space`
    catches un-edited templates.

    Args:
        config_class: A dataclass (not an instance) to introspect.

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


def generate_routing_templates() -> None:
    """Scaffold JSON search space templates from policy dataclass configs.

    Scans the ``logic.src.configs.policies`` package, finds dataclass
    ``*Config`` classes, and writes template JSON files to :data:`ROUTING_DIR`.
    Existing files are **not** overwritten to protect manually-tuned bounds.
    """
    from logic.src.configs import policies as policies_pkg

    os.makedirs(ROUTING_DIR, exist_ok=True)
    package_path = policies_pkg.__path__

    for _, module_name, is_pkg in pkgutil.walk_packages(package_path):
        if is_pkg or module_name in ("abc", "__init__"):
            continue

        full_module_name = f"logic.src.configs.policies.{module_name}"
        try:
            module = importlib.import_module(full_module_name)
        except Exception as exc:
            print(f"[generate_routing_templates] Error importing {full_module_name}: {exc}")
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
        output_path = os.path.join(ROUTING_DIR, f"{base_name}.json")

        if os.path.exists(output_path):
            print(f"[generate_routing_templates] Skipping existing file: {output_path}")
            continue

        params = _extract_params_from_config(config_class)
        with open(output_path, "w") as fh:
            json.dump(params, fh, indent=4)
        print(f"[generate_routing_templates] Generated: {output_path}")


def generate_route_improvement_templates() -> None:
    """Scaffold JSON search space templates for route improvement configs.

    Scans ``logic.src.configs.policies.route_improvement`` and writes
    template JSON files to :data:`ROUTE_IMPROVEMENT_DIR`.  Existing files are not
    overwritten.
    """
    from logic.src.configs.policies import route_improvement

    os.makedirs(ROUTE_IMPROVEMENT_DIR, exist_ok=True)

    for name, obj in inspect.getmembers(route_improvement):
        if not (
            (name.endswith("Config") or name.endswith("PostConfig"))
            and dataclasses.is_dataclass(obj)
            and getattr(obj, "__module__", "") == route_improvement.__name__
        ):
            continue

        base_name = name.lower().replace("postconfig", "").replace("config", "")
        output_path = os.path.join(ROUTE_IMPROVEMENT_DIR, f"{base_name}.json")

        if os.path.exists(output_path):
            print(f"[generate_route_improvement_templates] Skipping existing: {output_path}")
            continue

        params = _extract_params_from_config(obj)
        with open(output_path, "w") as fh:
            json.dump(params, fh, indent=4)
        print(f"[generate_route_improvement_templates] Generated: {output_path}")


def generate_packing_templates() -> None:
    """Scaffold JSON search space templates for packing configs.

    Scans the modules that define the packing configs (``packing_dispatch``,
    ``packing_learned`` and ``packing_threshold``) and
    writes template JSON files to :data:`PACKING_DIR`.  Existing files are not
    overwritten.
    """
    from logic.src.configs.policies.packing import dispatch as packing_dispatch
    from logic.src.configs.policies.packing import learned as packing_learned
    from logic.src.configs.policies.packing import threshold as packing_threshold

    os.makedirs(PACKING_DIR, exist_ok=True)

    members = [
        (name, obj)
        for module in (packing_dispatch, packing_learned, packing_threshold)
        for name, obj in inspect.getmembers(module)
        if getattr(obj, "__module__", "") == module.__name__  # each config once, from its defining module
    ]
    for name, obj in sorted(members, key=lambda item: item[0]):
        if not (name.endswith("Config") and dataclasses.is_dataclass(obj)):
            continue

        base_name = name.lower().replace("selectionconfig", "").replace("postconfig", "").replace("config", "")
        output_path = os.path.join(PACKING_DIR, f"{base_name}.json")

        if os.path.exists(output_path):
            print(f"[generate_packing_templates] Skipping existing: {output_path}")
            continue

        params = _extract_params_from_config(obj)
        with open(output_path, "w") as fh:
            json.dump(params, fh, indent=4)
        print(f"[generate_packing_templates] Generated: {output_path}")


def generate_acceptance_criteria_templates() -> None:
    """Scaffold JSON search space templates for acceptance criteria configs.

    Scans ``logic.src.configs.policies.acceptance_criteria`` and writes
    template JSON files to :data:`ACCEPTANCE_CRITERIA_DIR`. Existing files are not
    overwritten.
    """
    from logic.src.configs.policies import acceptance_criteria

    os.makedirs(ACCEPTANCE_CRITERIA_DIR, exist_ok=True)

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
        output_path = os.path.join(ACCEPTANCE_CRITERIA_DIR, f"{base_name}.json")

        if os.path.exists(output_path):
            print(f"[generate_acceptance_criteria_templates] Skipping existing: {output_path}")
            continue

        params = _extract_params_from_config(obj)
        with open(output_path, "w") as fh:
            json.dump(params, fh, indent=4)
        print(f"[generate_acceptance_criteria_templates] Generated: {output_path}")
