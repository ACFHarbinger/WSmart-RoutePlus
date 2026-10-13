"""
Utility to aggregate simulation results JSONs into a single Excel log.

Attributes:
    _DIST_PATTERN: Regex pattern to extract distribution information from policy names.
    _DISPLAY_METRICS: List of metrics to display in the summary.
    SELECTED_DIRS: Whitelist of directories to include in the summary.

Example:
    >>> from logic.src.utils.output.excel_summary import discover_and_aggregate
    >>> df = discover_and_aggregate()
    >>> print(df.head())
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from logic.src.constants import ROOT_DIR

# Constants
_DIST_PATTERN = re.compile(r"_(emp|gamma\d*|uniform)$", re.IGNORECASE)
_DISPLAY_METRICS = [
    "profit",
    "cost",
    "kg",
    "km",
    "kg/km",
    "overflows",
    "ncol",
    "kg_lost",
    "days",
    "time",
]

# Whitelist of directories to include in the summary.
# If empty, all directories under outputs/simulations/ are included.
# Example: ["31_days/riomaior_104", "30_days/riomaior_104"]
SELECTED_DIRS: List[str] = [
    "31_days/riomaior_104",
    "30_days/riomaior_104",
]


def _parse_policy_name(raw_name: str) -> Tuple[str, str]:
    """
    Split a raw policy key into (base_name, distribution).

    Args:
        raw_name:  Raw policy key.

    Returns:
        Tuple[str, str]: A tuple containing the base name and distribution.
    """
    match = _DIST_PATTERN.search(raw_name)
    if match:
        dist = match.group(1).lower()
        base = raw_name[: match.start()].rstrip("_")
        return base, dist
    return raw_name, "unknown"


def _load_json(path: str) -> Any:
    """
    Load and parse a JSON file.

    Args:
        path:  Path to the JSON file.

    Returns:
        Any: The parsed JSON data.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def discover_and_aggregate() -> pd.DataFrame:
    """
    Find all log directories and aggregate results into a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the aggregated simulation results.
    """
    output_root = os.path.join(ROOT_DIR, "outputs", "simulations")
    all_rows: List[Dict[str, Any]] = []

    if not os.path.isdir(output_root):
        print(f"Directory not found: {output_root}")
        return pd.DataFrame()

    for root, _, files in os.walk(output_root):
        rel_path = os.path.relpath(root, output_root)

        # Skip if not in whitelist (if whitelist is provided)
        if SELECTED_DIRS and not any(rel_path.startswith(d) for d in SELECTED_DIRS):
            continue

        # Find per-policy log files: log_<policy>_<N>N.json
        pol_files = [f for f in files if f.startswith("log_") and f.endswith(".json")]
        if not pol_files:
            continue

        for pol_file in pol_files:
            pol_path = os.path.join(root, pol_file)
            pol_data = _load_json(pol_path)
            if not isinstance(pol_data, dict) or "mean" not in pol_data:
                continue

            # Extract policy name from filename: log_<policy>_<N>N.json → <policy>
            stem = pol_file[4:]  # strip "log_"
            # Remove trailing _<digits>N.json
            import re as _re

            stem = _re.sub(r"_\d+N\.json$", "", stem)
            policy_key = stem

            mean_data = pol_data["mean"]
            std_data = pol_data.get("std", {})

            if not isinstance(mean_data, dict):
                continue

            base_name, dist = _parse_policy_name(policy_key)
            row: Dict[str, Any] = {
                "SourceDir": rel_path,
                "Policy": base_name,
                "Distribution": dist,
                "Policy_Key": policy_key,
            }

            for m, v in mean_data.items():
                row[f"{m}_mean"] = v
                row[f"{m}_std"] = std_data.get(m, 0.0) if isinstance(std_data, dict) else 0.0

            all_rows.append(row)

    return pd.DataFrame(all_rows)
