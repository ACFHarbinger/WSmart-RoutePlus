"""
Configuration loading utilities.

This module provides functions for:
- Loading YAML configuration files.
- Loading XML configuration files and converting them to dictionaries.

Attributes:
    load_yaml_config: Loads a YAML configuration file.
    load_xml_config: Loads an XML configuration file and converts it to a dictionary.
    load_config: Loads a YAML or XML configuration file based on extension.

Example:
    load_yaml_config("config.yaml")
    load_xml_config("config.xml")
    load_config("config.yaml")
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Union

import yaml

from logic.src.interfaces.traversable import ITraversable


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """
    Loads a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        dict[str, Any]: The configuration dictionary.

    Raises:
        ValueError: If parsing fails.
    """
    with open(config_path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}") from e


def load_xml_config(config_path: str) -> dict[str, Any]:
    """
    Loads an XML configuration file and converts it to a dictionary.

    Args:
        config_path: Path to the XML file.

    Returns:
        dict[str, Any]: The configuration dictionary.

    Raises:
        ValueError: If parsing fails.
    """
    try:
        tree = ET.parse(config_path)
        root = tree.getroot()

        def xml_to_dict(
            element: ET.Element,
        ) -> Union[Dict[str, Any], List[Any], str, int, float]:
            """
            Helper to convert XML element to dictionary.

            Args:
                element: The XML element.

            Returns:
                dict or list or str/int/float: The parsed data.
            """
            # Helper to convert XML element to dictionary
            if len(element) == 0:
                try:
                    return int(element.text) if element.text is not None else ""
                except (ValueError, TypeError):
                    try:
                        return float(element.text) if element.text is not None else ""
                    except (ValueError, TypeError):
                        return element.text if element.text is not None else ""

            result: dict[str, Any] = {}
            for child in element:
                child_data = xml_to_dict(child)
                if child.tag in result:
                    if isinstance(result[child.tag], list):
                        result[child.tag].append(child_data)
                    else:
                        result[child.tag] = [result[child.tag], child_data]
                else:
                    result[child.tag] = child_data
            return result

        data = xml_to_dict(root)
        if root.tag == "config" and isinstance(data, ITraversable):
            return data
        return {root.tag: data}

    except ET.ParseError as e:
        raise ValueError(f"Error parsing XML file: {e}") from e


def load_config(config_path: str) -> dict[str, Any]:
    """
    Loads a YAML or XML configuration file based on extension.

    Args:
        config_path: Path to the configuration file.

    Returns:
        dict[str, Any]: The configuration dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported or parsing fails.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    _, ext = os.path.splitext(config_path)

    config: dict[str, Any] = {}
    if ext.lower() == ".yaml" or ext.lower() == ".yml":
        config = load_yaml_config(config_path)
        if config is None:
            config = {}
        return config

    elif ext.lower() == ".xml":
        config = load_xml_config(config_path)
        if config is None:
            config = {}
        return config

    else:
        raise ValueError(f"Unsupported config file extension: {ext}")
