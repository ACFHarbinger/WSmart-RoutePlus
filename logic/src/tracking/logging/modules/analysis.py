"""Analysis utilities for experiment and simulation logs.

ResultsDB lives in ``analysis_results``.

Slug parsing lives in ``analysis_slug``.

This module provides tools for aggregating, summarising, and reporting on
experimental results.  The primary entry point for post-processing is
:class:`ResultsDB` together with :func:`load_log_dict`, which recursively
walk the output tree and expose every simulation result as a queryable,
filterable pandas DataFrame row.

Attributes:
    ResultsDB: Queryable container for all loaded simulation results.
    load_log_dict: Walk output directories and return a populated ResultsDB.
    output_stats: Compute/persist mean and std back to per-policy JSON files.
    runs_per_policy: Count completed samples per solver policy.
    final_simulation_summary: Print a formatted per-policy summary table.

Example:
    >>> from logic.src.tracking.logging.modules.analysis import load_log_dict
    >>> db = load_log_dict("/path/to/project", "output")
    >>> db.filter(packing="lookahead", n_bins=100).mean_metrics()
"""

import contextlib
import json
import os
import re
import statistics
import threading
from typing import Any, Dict, List, Optional, Tuple, cast

import logic.src.constants as udef
from logic.src.tracking.logging.modules.storage import update_policy_log_section
from logic.src.tracking.logging.modules.summary_display import (
    daily_row_is_visible,
    display_per_policy_simulation_summary,
    display_simulation_summary_table,
    final_simulation_summary,
)
from logic.src.utils.input.files import compose_dirpath, read_json

from .analysis_results import (
    ResultsDB as ResultsDB,
)
from .analysis_slug import (
    _ACCEPTANCE_CRITERIA as _ACCEPTANCE_CRITERIA,
)
from .analysis_slug import (
    _MANDATORY_PREFIXES as _MANDATORY_PREFIXES,
)
from .analysis_slug import (
    _ROUTE_IMPROVERS as _ROUTE_IMPROVERS,
)
from .analysis_slug import (
    _ROUTING_ALGORITHM_ENGINES as _ROUTING_ALGORITHM_ENGINES,
)
from .analysis_slug import (
    _ROUTING_ALGORITHMS as _ROUTING_ALGORITHMS,
)
from .analysis_slug import (
    _parse_slug as _parse_slug,
)

# ── Policy slug component vocabularies (ordered longest → shortest) ────────────
# Used by _parse_slug to decompose a full slug into its four named parts.

# Empty by default — these are highly project-specific and must be supplied by
# the caller via load_log_dict / _parse_slug when present in slug names.




def load_log_dict(  # noqa: C901
    home_dir: str,
    output_dir: str = "output",
    *,
    mandatory_prefixes: Optional[List[str]] = None,
    routing_algorithms: Optional[List[str]] = None,
    route_improvers: Optional[List[str]] = None,
    routing_algorithm_engines: Optional[List[str]] = None,
    acceptance_criteria: Optional[List[str]] = None,
    load_samples: bool = False,
    load_daily: bool = False,
    lock: Optional[threading.Lock] = None,
) -> ResultsDB:
    """Walk all simulation output directories and return a queryable ResultsDB.

    Recursively scans ``{home_dir}/{output_dir}/`` for files matching
    ``log_*_<N>N.json`` (realtime JSONL files are skipped).  Each file is
    parsed into a record containing:

    * **Path metadata**: ``n_days``, ``n_bins``, ``area``, ``waste_type``,
      ``data_distribution``, ``run_name``.
    * **Policy components** (best-effort slug decomposition):
      ``packing``, ``routing_algorithm``, ``acceptance_criterion``,
      ``route_improver``, ``policy_slug``.
    * **Mean / std metrics**: read directly from the ``"mean"`` / ``"std"``
      sections of each JSON; computed from ``"samples"`` when those sections are
      absent.

    The expected path structure is::

        {home_dir}/{output_dir}/
          {n_days}days/
            {area}{n_bins}_{waste_type}/
              {data_distribution}/
                {run_name}/
                  log_{policy_slug}_{N}N.json

    Slug decomposition uses the module-level vocabulary constants by default.
    Pass explicit lists to override them when your result directories use
    different naming conventions — all three lists are ordered longest-first
    so that longer tokens shadow shorter ones that share a prefix::

        db = load_log_dict(
            home_dir,
            mandatory_prefixes=["my_custom_strat", "fallback"],
            routing_algorithms=["vrpp_gurobi", "gurobi", "my_solver"],
            route_improvers=["tsp_fast_tsp", "ftsp", "tsp"],
        )

    Args:
        home_dir: Project root.
        output_dir: Results tree, relative to ``home_dir`` (or absolute).
        mandatory_prefixes: Override the default ``_MANDATORY_PREFIXES`` list
            used by slug parsing.  Pass ``None`` to keep the built-in defaults.
        routing_algorithms: Override the default ``_ROUTING_ALGORITHMS`` list.
            Pass ``None`` to keep the built-in defaults.
        route_improvers: Override the default ``_ROUTE_IMPROVERS`` list.
            Pass ``None`` to keep the built-in defaults.
        routing_algorithm_engines: Override the default
            ``_ROUTING_ALGORITHM_ENGINES`` list (empty by default — must be
            supplied when routing algorithm engines appear in slug names).
        acceptance_criteria: Override the default ``_ACCEPTANCE_CRITERIA`` list
            (empty by default — whatever remains after all other components are
            stripped becomes the acceptance criterion).
        load_samples: Attach the raw ``"samples"`` section to each record.
        load_daily: Attach the raw ``"daily"`` section to each record.
        lock: Optional thread lock for concurrent file access.

    Returns:
        ResultsDB: Use ``.filter()``, ``.groupby()``, and ``.to_dataframe()``
        to access and slice the data.

    Example::

        db = load_log_dict(home_dir)
        # All 100-bin lookahead results for the emp distribution
        sub = db.filter(n_bins=100, packing="lookahead",
                        data_distribution="emp")
        print(sub.mean_metrics())
        # Compare profit across routing algorithms
        sub.groupby("routing_algorithm")["profit"].mean().sort_values()
    """
    import glob as _glob

    base_dir = os.path.join(home_dir, output_dir)
    records: List[Dict[str, Any]] = []

    for fpath in sorted(_glob.glob(os.path.join(base_dir, "**", "log_*N.json"), recursive=True)):
        fname = os.path.basename(fpath)
        if fname.startswith("log_realtime"):
            continue

        # Filename → slug + sample count
        m_file = re.match(r"^log_(.+?)_(\d+)N\.json$", fname)
        if not m_file:
            continue
        slug, n_samples_val = m_file.group(1), int(m_file.group(2))

        # Path → (n_days, area, n_bins, waste_type, data_dist, run_name)
        rel = os.path.relpath(fpath, base_dir)
        parts = rel.replace("\\", "/").split("/")
        if len(parts) < 3:
            continue

        m_days = re.match(r"^(\d+)days$", parts[0])
        if not m_days:
            continue
        n_days_val = int(m_days.group(1))

        # Area is all letters; n_bins follows immediately; optional _waste_type
        m_area = re.match(r"^([a-zA-Z]+)(\d+)(?:_(.*))?$", parts[1])
        if not m_area:
            continue
        area_str = m_area.group(1)
        n_bins_val = int(m_area.group(2))
        waste_type_str = m_area.group(3) or ""

        # parts between area-dir and filename → [dist, run_name_parts…]
        inner = parts[2:-1]
        data_dist = inner[0] if inner else ""
        run_name_str = "/".join(inner[1:]) if len(inner) > 1 else ""

        # Load JSON
        try:
            pol_data = cast(Dict[str, Any], read_json(fpath, lock))
        except Exception:
            continue
        if not isinstance(pol_data, dict):
            continue

        mean_dict: Dict[str, Any] = pol_data.get("mean") or {}
        std_dict: Dict[str, Any] = pol_data.get("std") or {}

        # Compute mean/std from samples if the pre-computed sections are absent
        if not mean_dict and "samples" in pol_data:
            s_sec = pol_data["samples"]
            if isinstance(s_sec, dict):
                raw_vals = [
                    list(v.values()) if isinstance(v, dict) else list(v)
                    for v in s_sec.values()
                    if isinstance(v, (dict, list))
                ]
                if raw_vals:
                    first_val = next(iter(s_sec.values()))
                    s_keys: List[str] = list(first_val.keys()) if isinstance(first_val, dict) else []
                    zipped = list(zip(*raw_vals, strict=False))
                    mean_dict = {k: statistics.mean(c) for k, c in zip(s_keys, zipped, strict=False)}
                    if len(raw_vals) > 1:
                        std_dict = {k: statistics.stdev(c) for k, c in zip(s_keys, zipped, strict=False)}

        record: Dict[str, Any] = {
            "n_days": n_days_val,
            "n_bins": n_bins_val,
            "area": area_str,
            "waste_type": waste_type_str,
            "data_distribution": data_dist,
            "run_name": run_name_str,
            "policy_slug": slug,
            "n_samples": n_samples_val,
            **_parse_slug(
                slug,
                mandatory_prefixes,
                routing_algorithms,
                route_improvers,
                routing_algorithm_engines,
                acceptance_criteria,
            ),
        }
        for k, v in mean_dict.items():
            record[k] = v
        for k, v in std_dict.items():
            record[k + "_std"] = v
        if load_samples:
            record["samples"] = pol_data.get("samples")
        if load_daily:
            record["daily"] = pol_data.get("daily")

        records.append(record)

    return ResultsDB.from_records(records)


@compose_dirpath
def output_stats(  # noqa: C901
    dir_path: str,
    nsamples: int,
    policies: List[str],
    keys: List[str],
    sort_log_func: Optional[Any] = None,
    print_output: bool = False,
    lock: Optional[threading.Lock] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute mean and std statistics for policies and write to JSON.

    Reads per-sample data from ``log_{pol}_{nsamples}N.json`` files (``samples``
    section) and writes back the computed ``mean`` and ``std`` sections.

    Args:
        dir_path: Directory containing per-policy log JSON files.
        nsamples: Number of simulation samples in the full log.
        policies: List of policy names to include in statistics.
        keys: Metric names (e.g. 'profit', 'km') to aggregate.
        sort_log_func: Optional function to reorder entries. Defaults to None.
        print_output: Whether to print results to stdout. Defaults to False.
        lock: Optional thread lock for concurrent file access. Defaults to None.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Mean dict and Std-dev dict.
    """
    mean_dit: Dict[str, Any] = {}
    std_dit: Dict[str, Any] = {}

    acquired = lock.acquire(timeout=udef.LOCK_TIMEOUT) if lock is not None else True
    if not acquired:
        return {}, {}
    try:
        suffix = f"_{nsamples}N.json"
        # When no policies specified, auto-discover all log files in the directory.
        effective_policies: List[str] = list(policies)
        if not effective_policies and os.path.isdir(dir_path):
            for _fname in sorted(os.listdir(dir_path)):
                if _fname.startswith("log_") and _fname.endswith(suffix):
                    effective_policies.append(_fname[4 : _fname.rfind(suffix)])

        for pol in effective_policies:
            pol_log_path = os.path.join(dir_path, f"log_{pol}{suffix}")
            if not os.path.isfile(pol_log_path):
                # Fallback: substring scan so short display names (e.g. "aco_hh") resolve
                # to full-slug filenames (e.g. log_lookahead_aco_hh_custom_ftsp_1N.json).
                pol_log_path = None  # type: ignore[assignment]
                if os.path.isdir(dir_path):
                    for _fname in os.listdir(dir_path):
                        if _fname.startswith("log_") and _fname.endswith(suffix):
                            _slug = _fname[4 : _fname.rfind(suffix)]
                            if pol in _slug:
                                pol_log_path = os.path.join(dir_path, _fname)
                                break
                if pol_log_path is None:
                    continue
            try:
                pol_data = cast(Dict[str, Any], read_json(pol_log_path, lock=None))
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(pol_data, dict) or "samples" not in pol_data:
                continue
            samples_section = pol_data["samples"]
            # samples_section is {str(sample_id): {metric: val}}
            sample_values = [list(v.values()) for v in samples_section.values() if isinstance(v, dict)]
            if not sample_values:
                continue
            mean_vals = [*map(statistics.mean, zip(*sample_values, strict=False))]
            mean_dit[pol] = dict(zip(keys, mean_vals, strict=False))
            if len(sample_values) > 1:
                std_vals = [*map(statistics.stdev, zip(*sample_values, strict=False))]
                std_dit[pol] = dict(zip(keys, std_vals, strict=False))
            else:
                std_dit[pol] = {key: 0.0 for key in keys}

            if mean_dit.get(pol):  # never overwrite with an empty dict
                update_policy_log_section(pol_log_path, "mean", mean_dit[pol], lock=None)
                update_policy_log_section(pol_log_path, "std", std_dit[pol], lock=None)

        if sort_log_func:
            mean_dit = sort_log_func(mean_dit)
            std_dit = sort_log_func(std_dit)

        if print_output:
            for pol in mean_dit:
                print(f"{pol}:")
                for key in keys:
                    m = mean_dit[pol].get(key, 0.0)
                    s = std_dit[pol].get(key, 0.0) if pol in std_dit else 0.0
                    print(f"- {key}: {m:.2f} +- {s:.4f}")
    finally:
        if lock is not None:
            lock.release()
    return mean_dit, std_dit


@compose_dirpath
def runs_per_policy(  # noqa: C901
    dir_paths: List[str],
    nsamples: List[int],
    policies: List[str],
    print_output: bool = False,
    lock: Optional[threading.Lock] = None,
) -> List[Dict[str, List[int]]]:
    """Count runs per policy from full log files.

    Args:
        dir_paths: List of output directories to inspect.
        nsamples: Expected sample counts for each directory.
        policies: List of policy names to search for.
        print_output: Whether to print counts to stdout. Defaults to False.
        lock: Optional thread lock for path scanning. Defaults to None.

    Returns:
        List[Dict[str, List[int]]]: For each path, a map of policy to sample IDs found.
    """
    runs_ls = []
    for path, ns in zip(dir_paths, nsamples, strict=False):
        dit: Dict[str, List[int]] = {pol: [] for pol in policies}
        _suffix = f"_{ns}N.json"
        for pol in policies:
            pol_log_path = os.path.join(path, f"log_{pol}{_suffix}")
            if not os.path.exists(pol_log_path):
                # Fallback: substring scan for display-name style policies.
                pol_log_path = None  # type: ignore[assignment]
                if os.path.isdir(path):
                    for _fname in os.listdir(path):
                        if _fname.startswith("log_") and _fname.endswith(_suffix):
                            _slug = _fname[4 : _fname.rfind(_suffix)]
                            if pol in _slug:
                                pol_log_path = os.path.join(path, _fname)
                                break
                if pol_log_path is None:
                    continue
            try:
                pol_data = cast(Dict[str, Any], read_json(pol_log_path, lock))
            except Exception:
                continue
            if isinstance(pol_data, dict) and "samples" in pol_data:
                for sample_id_str in pol_data["samples"]:
                    with contextlib.suppress(ValueError, TypeError):
                        dit[pol].append(int(sample_id_str))
        runs_ls.append(dit)
        if print_output:
            # Extract nbins from the path component that follows the "<ndays>days" segment.
            parts = path.replace("\\", "/").split("/")
            gsize = 0
            for _i, _p in enumerate(parts):
                if re.match(r"^\d+days$", _p) and _i + 1 < len(parts):
                    _m = re.search(r"(\d+)", parts[_i + 1])
                    if _m:
                        gsize = int(_m.group(1))
                    break
            print(f"graph {gsize} #runs per policy:")
            for key, val in dit.items():
                print(f"- {key}: {len(val)} samples: {val}")
    return runs_ls


__all__ = [
    "ResultsDB",
    "load_log_dict",
    "output_stats",
    "runs_per_policy",
    "final_simulation_summary",
    "display_simulation_summary_table",
    "daily_row_is_visible",
    "display_per_policy_simulation_summary",
]
