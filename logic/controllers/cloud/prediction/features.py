"""Features of a cell for runtime and memory prediction (CR-11).

A cell is described by its routing algorithm, packing variant, improver, demand
distribution, network (area and size), horizon and the machine type it runs on.
They are read from the cell's ``test_sim`` arguments first and from its id second;
rerun manifests use ids of the form ``<H>d/<area>/<size>/<dist>/<run_name>/<variant>/<policy>``,
where ``run_name`` ends with the improver (``la_cls``, ``sl_ftsp``).

Attributes:
    CATEGORICAL: Categorical feature names.
    NUMERIC: Numeric feature names.
    cell_features: Feature dict of a :class:`CellSpec`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from logic.controllers.cloud.model import CellSpec

CATEGORICAL = ["policy", "packing", "improver", "distribution", "area", "machine_type"]
NUMERIC = ["size", "horizon"]
UNKNOWN = "unknown"
_IMPROVERS = ("cls", "ftsp", "none")
_PACKING = ("lookahead", "cf70", "cf90", "sl1", "sl2")


def _argv_values(argv: List[str]) -> Dict[str, str]:
    values = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if sep:
            values[key.lstrip("+")] = value.strip("'\"")
    return values


def _policy(values: Dict[str, str], parts: List[str]) -> str:
    match = re.fullmatch(r"\[\s*([^\],\s]+)\s*\]", values.get("sim.policies", ""))
    if match:
        return match.group(1).strip("'\"")
    return parts[-1] if len(parts) >= 7 else UNKNOWN


def cell_features(cell: CellSpec, machine_type: Optional[str] = None) -> Dict[str, Any]:
    """The prediction features of ``cell`` (unknown values become ``"unknown"`` or None)."""
    values = _argv_values(cell.argv)
    parts = cell.cell_id.split("/")
    run_name = values.get("sim.run_name") or (parts[4] if len(parts) >= 7 else "")
    improver = next((i for i in _IMPROVERS if run_name.endswith(i)), UNKNOWN)
    packing = parts[5] if len(parts) >= 7 and parts[5] in _PACKING else UNKNOWN

    def number(key: str, index: int, suffix: str = "") -> Optional[int]:
        raw = values.get(key) or (parts[index].removesuffix(suffix) if len(parts) > index else "")
        return int(raw) if raw.isdigit() else None

    return {
        "policy": _policy(values, parts),
        "packing": packing,
        "improver": improver,
        "distribution": values.get("sim.data_distribution") or (parts[3] if len(parts) >= 7 else UNKNOWN),
        "area": values.get("sim.graph.area") or (parts[1] if len(parts) >= 7 else UNKNOWN),
        "size": number("sim.graph.num_loc", 2),
        "horizon": number("sim.graph.n_days", 0, "d"),
        "machine_type": machine_type or UNKNOWN,
    }
