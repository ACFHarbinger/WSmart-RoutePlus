"""
Utilities to convert YAML configuration files to environment variables.

Attributes:
    to_bash_value: Convert a Python value to a Bash-friendly string representation.
    load_yaml_env: Load a YAML configuration file and recursively merge its defaults.
    deep_merge: Deeply merge source dictionary into target dictionary.
    main: Main entry point to convert YAML files to environment variables.

Example:
    to_bash_value(value)
    load_yaml_env(config_path)
    deep_merge(target, source)
"""

import os
import sys
from typing import Any

import yaml

from logic.src.interfaces.traversable import ITraversable


def to_bash_value(value: Any) -> str:
    """Convert a Python value to a Bash-friendly string representation.

    Args:
        value: The value to convert.

    Returns:
        str: Bash-friendly string representation.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, ITraversable):
        # Convert dict to bash associative array: ( ["key1"]="val1" ["key2"]="val2" )
        # Note: Bash 4.0+ required.
        entries = []
        for k, v in value.items():
            entries.append(f'["{str(k)}"]="{str(v)}"')
        return f"({' '.join(entries)})"
    elif isinstance(value, list):
        # Convert list to bash array: ("item1" "item2")
        # Quote each item to handle spaces
        items = [f'"{str(v)}"' for v in value]
        return f"({' '.join(items)})"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        # String, quote it
        return f'"{str(value)}"'


def load_yaml_env(config_path: str) -> dict:
    """Load a YAML configuration file and recursively merge its defaults.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        dict: Merged configuration dictionary.
    """
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML {config_path}: {e}", file=sys.stderr)
            sys.exit(1)

    if "defaults" not in config:
        return config

    defaults = config.pop("defaults")
    final_merged = {}  # type: ignore[var-annotated]

    config_dir = os.path.dirname(os.path.abspath(config_path))
    # If we are in a subdirectory (like tasks/), the base config dir is the parent
    if os.path.basename(config_dir) in ["models", "envs", "data", "tasks"]:
        config_dir = os.path.dirname(config_dir)

    for d in defaults:
        if d == "_self_":
            final_merged.update(config)
        elif isinstance(d, ITraversable):
            for folder, filename in d.items():
                if filename:
                    # Look for file in the same configs/ folder structure
                    sub_config_path = os.path.join(config_dir, folder, f"{filename}.yaml")
                    if os.path.exists(sub_config_path):
                        sub_config = load_yaml_env(sub_config_path)
                        final_merged.update(sub_config)
                    else:
                        print(f"Warning: Default config {sub_config_path} not found.", file=sys.stderr)
        elif isinstance(d, str):
            # Handle pure strings if needed, maybe as direct siblings or full paths
            pass

    # Merge remaining keys from config that might not have been merged via _self_
    final_merged.update(config)
    return final_merged


def deep_merge(target: dict, source: dict) -> dict:
    """
    Deeply merge source dictionary into target dictionary.

    Args:
        target: Target dictionary
        source: Source dictionary

    Returns:
        Merged dictionary
    """
    for key, value in source.items():
        if (
            isinstance(value, ITraversable)
            and target.get(key) is not None
            and isinstance(target.get(key), ITraversable)
        ):
            deep_merge(target[key], value) # pyrefly: ignore [bad-argument-type]
        else:
            target[key] = value
    return target


def main() -> None:
    """
    Main entry point to convert YAML files to environment variables.

    Args:
        None

    """
    if len(sys.argv) < 2:
        print("Usage: python yaml_to_env.py <config1.yaml> [<config2.yaml> ...]", file=sys.stderr)
        sys.exit(1)

    final_config = {}  # type: ignore[var-annotated]

    for config_path in sys.argv[1:]:
        config = load_yaml_env(config_path)
        deep_merge(final_config, config)

    def flatten_dict(d, parent_key="", sep="_"):
        """Flatten a nested dictionary into single-level keys with separators."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, ITraversable):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Export top-level keys as before
    for key, value in final_config.items():
        bash_var_name = key.upper()
        bash_value = to_bash_value(value)
        if isinstance(value, ITraversable):
            # Associative arrays must be declared with -A
            print(f"declare -A {bash_var_name}={bash_value}")
        else:
            print(f"export {bash_var_name}={bash_value}")

    # Export flattened keys for nested structures
    flattened_config = flatten_dict(final_config)
    for key, value in flattened_config.items():
        # Avoid re-exporting top-level keys already handled
        if "_" not in key:
            continue
        bash_var_name = key.upper()
        bash_value = to_bash_value(value)
        # We don't export nested dicts as flattened variables here, only leaves
        if not isinstance(value, ITraversable):
            print(f"export {bash_var_name}={bash_value}")


if __name__ == "__main__":
    main()
