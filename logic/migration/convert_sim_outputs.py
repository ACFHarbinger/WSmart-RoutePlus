"""Convert archived simulator logs to the WP-08 day-record schema.

Maps the old ``days`` column to ``collection_days`` and derives
``days_simulated`` from the run config when that file is given, otherwise
from the number of daily records. Drops ``reward``. Writes a new tree.
Does not modify the input.

This tool is intentionally not run by the refactor. Point it at an explicit
input and output directory when an archive should be converted.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_days_simulated(config_path: Optional[str]) -> Optional[float]:
    """Read the horizon from a small JSON or YAML run config, if one was given."""
    if not config_path:
        return None
    with open(config_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if config_path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - pyyaml is a project dependency
            raise RuntimeError("PyYAML is required to read a YAML run config") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Run config must be a mapping: {config_path}")
    sim = payload.get("sim", payload)
    if isinstance(sim, dict) and "days" in sim:
        return float(sim["days"])
    graph = sim.get("graph", {}) if isinstance(sim, dict) else {}
    if isinstance(graph, dict) and "n_days" in graph:
        return float(graph["n_days"])
    raise ValueError("Run config has no sim.days or sim.graph.n_days")


def convert_record(record: Dict[str, Any], days_simulated: Optional[float], scope: str = "summary") -> Dict[str, Any]:
    """Return one sample or day record in the new schema."""
    out: Dict[str, Any] = {}
    for key, value in record.items():
        if key == "reward":
            continue
        out[key] = _convert_payload(value, days_simulated, key if key in ("daily", "std") else scope)
    if "days" in out and "collection_days" not in out:
        out["collection_days"] = out.pop("days")
    elif "days" in out:
        out.pop("days")
    if scope == "daily":
        # Daily column arrays and row records have different shapes; neither
        # should receive the whole run's horizon on every day.
        day_column = record.get("day")
        if isinstance(day_column, list):
            out["days_simulated"] = [1.0] * len(day_column)
            tours = record.get("tour")
            if isinstance(tours, list) and len(tours) == len(day_column):
                out["collection_days"] = [float(any(node != 0 for node in tour)) for tour in tours]
        else:
            out["days_simulated"] = 1.0
            if isinstance(record.get("tour"), list):
                out["collection_days"] = float(any(node != 0 for node in record["tour"]))
        return out
    stamp_horizon = "days" in record or "kg" in record or "profit" in record
    if "days_simulated" not in out and stamp_horizon:
        if days_simulated is not None:
            out["days_simulated"] = 0.0 if scope == "std" else days_simulated
        elif isinstance(out.get("daily"), list):
            out["days_simulated"] = float(len(out["daily"]))
        elif isinstance(record.get("daily"), dict):
            kg = record["daily"].get("kg") or record["daily"].get("day") or []
            out["days_simulated"] = float(len(kg))
    return out


def convert_tree(source: str, dest: str, days_simulated: Optional[float]) -> int:
    """Copy JSON logs from ``source`` to ``dest`` with converted records.

    Returns the number of JSON files written.
    """
    source_path, dest_path = Path(source).resolve(), Path(dest).resolve()
    if source_path == dest_path or source_path in dest_path.parents or dest_path in source_path.parents:
        raise ValueError("Input and output trees must not overlap")
    if not source_path.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")
    written = 0
    for dirpath, _dirnames, filenames in os.walk(source):
        for name in filenames:
            if not name.endswith(".json"):
                continue
            src_path = os.path.join(dirpath, name)
            rel = os.path.relpath(src_path, source)
            dst_path = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(src_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            converted = _convert_payload(payload, days_simulated)
            with open(dst_path, "w", encoding="utf-8") as handle:
                json.dump(converted, handle, indent=2)
                handle.write("\n")
            written += 1
    return written


def _convert_payload(payload: Any, days_simulated: Optional[float], scope: str = "summary") -> Any:
    if isinstance(payload, list):
        return [_convert_payload(item, days_simulated, scope) for item in payload]
    if isinstance(payload, dict):
        if any(key in payload for key in ("reward", "days", "kg", "profit", "daily")):
            return convert_record(payload, days_simulated, scope)
        return {
            key: _convert_payload(value, days_simulated, key if key in ("daily", "std") else scope)
            for key, value in payload.items()
        }
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry. Refuses to guess an archive directory."""
    parser = argparse.ArgumentParser(description="Convert archived simulator JSON to the WP-08 schema.")
    parser.add_argument("--input", required=True, help="Directory of archived JSON logs")
    parser.add_argument("--output", required=True, help="Directory to write converted JSON logs")
    parser.add_argument("--run-config", default=None, help="JSON or YAML with sim.days or sim.graph.n_days")
    args = parser.parse_args(argv)
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise SystemExit("input and output must be different directories")
    days = _load_days_simulated(args.run_config)
    count = convert_tree(args.input, args.output, days)
    print(f"converted {count} json file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
