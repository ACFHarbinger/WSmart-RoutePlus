"""Analysis utilities for experiment and simulation logs.

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
    >>> db.filter(mandatory_selection="lookahead", n_bins=100).mean_metrics()
"""

import contextlib
import json
import os
import re
import statistics
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

import logic.src.constants as udef
from logic.src.tracking.logging.modules.storage import update_policy_log_section
from logic.src.utils.input.files import compose_dirpath, read_json

# ── Policy slug component vocabularies (ordered longest → shortest) ────────────
# Used by _parse_slug to decompose a full slug into its four named parts.

_MANDATORY_PREFIXES: List[str] = [
    "last_minute_cf70",
    "last_minute_cf80",
    "last_minute_cf90",
    "last_minute_cf",
    "service_level1",
    "service_level2",
    "service_level",
    "last_minute",
    "regular_lvl5",
    "regular_lvl4",
    "regular_lvl3",
    "regular_lvl2",
    "regular_lvl1",
    "regular",
    "lookahead",
]

_ROUTE_IMPROVERS: List[str] = [
    "tsp_fast_tsp",
    "cvrp_ortools",
    "ftsp",
    "tsp",
]

_ROUTE_CONSTRUCTORS: List[str] = [
    "vrpp_gurobi",
    "pg_clns",
    "swc_tcf",
    "sans_new",
    "ks_aco",
    "aco_hh",
    "alns",
    "psoma",
    "sisr",
    "hvpl",
    "gurobi",
    "ortools",
    "sans",
    "hgs",
    "bpc",
]

# Empty by default — these are highly project-specific and must be supplied by
# the caller via load_log_dict / _parse_slug when present in slug names.
_ROUTE_CONSTRUCTOR_ENGINES: List[str] = []

_ACCEPTANCE_CRITERIA: List[str] = []


def _parse_slug(
    slug: str,
    mandatory_prefixes: Optional[List[str]] = None,
    route_constructors: Optional[List[str]] = None,
    route_improvers: Optional[List[str]] = None,
    route_constructor_engines: Optional[List[str]] = None,
    acceptance_criteria: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Best-effort decomposition of a policy slug into its five named components.

    The expected slug format is::

        {mandatory_selection}_{route_constructor}_{route_constructor_engine}_{acceptance_criterion}_{route_improver}

    Any component may be absent.  Parsing proceeds in this order so that each
    vocabulary is applied to the right portion of the slug:

    1. Strip the ``mandatory_selection`` prefix (longest match from the start).
    2. Strip the ``route_improver`` suffix (longest match from the end).
    3. Strip the ``route_constructor`` prefix (longest match from the start).
    4. Strip the ``route_constructor_engine`` prefix (longest match from the
       start of the remainder).
    5. Match ``acceptance_criterion`` from the vocabulary (longest prefix match);
       fall back to consuming the whole remainder when no vocabulary entry
       matches.

    Vocabulary lists default to the module-level constants when ``None``
    is passed.  ``_ROUTE_CONSTRUCTOR_ENGINES`` and ``_ACCEPTANCE_CRITERIA``
    default to empty lists, so callers must supply them when their slugs
    contain those components.

    Args:
        slug: Full policy slug, e.g.
            ``"lookahead_aco_hh_ortools_custom_ftsp"``.
        mandatory_prefixes: Ordered mandatory-selection prefixes (longest-first).
            Defaults to ``_MANDATORY_PREFIXES``.
        route_constructors: Ordered route-constructor names.  Defaults to
            ``_ROUTE_CONSTRUCTORS``.
        route_improvers: Ordered route-improver suffixes (longest-first).
            Defaults to ``_ROUTE_IMPROVERS``.
        route_constructor_engines: Ordered constructor-engine names
            (longest-first).  Defaults to ``_ROUTE_CONSTRUCTOR_ENGINES``
            (empty — must be supplied when engines appear in slugs).
        acceptance_criteria: Ordered acceptance-criterion names (longest-first).
            Defaults to ``_ACCEPTANCE_CRITERIA`` (empty — whatever remains
            after all other components are stripped becomes the criterion).

    Returns:
        Dict with keys ``mandatory_selection``, ``route_constructor``,
        ``route_constructor_engine``, ``acceptance_criterion``, and
        ``route_improver``.  Missing components are empty strings.
    """
    m_prefixes = mandatory_prefixes if mandatory_prefixes is not None else _MANDATORY_PREFIXES
    r_constructors = route_constructors if route_constructors is not None else _ROUTE_CONSTRUCTORS
    r_improvers = route_improvers if route_improvers is not None else _ROUTE_IMPROVERS
    r_engines = route_constructor_engines if route_constructor_engines is not None else _ROUTE_CONSTRUCTOR_ENGINES
    a_criteria = acceptance_criteria if acceptance_criteria is not None else _ACCEPTANCE_CRITERIA

    remainder = slug

    # 1. Strip mandatory-selection prefix from the start.
    mandatory = ""
    for prefix in m_prefixes:
        if remainder.startswith(prefix + "_"):
            mandatory = prefix
            remainder = remainder[len(prefix) + 1 :]
            break

    # 2. Strip route-improver suffix from the end.
    improver = ""
    for suf in r_improvers:
        if remainder.endswith("_" + suf):
            improver = suf
            remainder = remainder[: -(len(suf) + 1)]
            break

    # 3. Strip route-constructor prefix from the start.
    constructor = ""
    for c in r_constructors:
        if remainder.startswith(c):
            constructor = c
            rest = remainder[len(c) :]
            remainder = rest[1:] if rest.startswith("_") else rest
            break

    # 4. Strip constructor engine from the start of what remains.
    engine = ""
    for e in r_engines:
        if remainder.startswith(e):
            engine = e
            rest = remainder[len(e) :]
            remainder = rest[1:] if rest.startswith("_") else rest
            break

    # 5. Match acceptance criterion from vocabulary; fall back to the whole remainder.
    acceptance = ""
    for a in a_criteria:
        if remainder.startswith(a):
            rest = remainder[len(a) :]
            # Only consume if it accounts for the whole remainder or is followed by '_'.
            if rest == "" or rest.startswith("_"):
                acceptance = a
                remainder = rest[1:] if rest.startswith("_") else ""
                break
    if not acceptance:
        acceptance = remainder

    return {
        "mandatory_selection": mandatory,
        "route_constructor": constructor,
        "route_constructor_engine": engine,
        "acceptance_criterion": acceptance,
        "route_improver": improver,
    }


class ResultsDB:
    """Queryable container for simulation results loaded from log directories.

    Each row in the underlying DataFrame represents the aggregated results for
    one policy / graph-size / distribution / run_name combination.  The
    ``filter`` method narrows the view without modifying the original; calls
    can be chained.

    Attributes:
        METADATA_COLS: Descriptive column names that identify a result's context.

    Example::

        db = load_log_dict(home_dir, "output")
        # Filter then view as DataFrame
        df = db.filter(mandatory_selection="lookahead", n_bins=100).mean_metrics()
        # Group by constructor, average profit across graph sizes
        db.to_dataframe().groupby("route_constructor")["profit"].mean()
    """

    METADATA_COLS: List[str] = [
        "n_days",
        "n_bins",
        "area",
        "waste_type",
        "data_distribution",
        "run_name",
        "policy_slug",
        "n_samples",
        "mandatory_selection",
        "route_constructor",
        "route_constructor_engine",
        "acceptance_criterion",
        "route_improver",
    ]

    def __init__(self, df: Any) -> None:
        self._df = df

    @classmethod
    def from_records(cls, records: List[Dict[str, Any]]) -> "ResultsDB":
        """Construct a ResultsDB from a list of flat record dicts."""
        try:
            import pandas as pd

            return cls(pd.DataFrame(records))
        except ImportError as exc:
            raise ImportError("pandas is required for ResultsDB") from exc

    # ── Filtering ─────────────────────────────────────────────────────────────

    def filter(
        self,
        n_days: Optional[Any] = None,
        n_bins: Optional[Any] = None,
        area: Optional[str] = None,
        waste_type: Optional[str] = None,
        data_distribution: Optional[str] = None,
        run_name: Optional[str] = None,
        policy_slug: Optional[str] = None,
        mandatory_selection: Optional[str] = None,
        route_constructor: Optional[str] = None,
        route_constructor_engine: Optional[str] = None,
        acceptance_criterion: Optional[str] = None,
        route_improver: Optional[str] = None,
    ) -> "ResultsDB":
        """Return a narrowed ResultsDB matching all non-None filters.

        String parameters use case-insensitive substring matching.
        Numeric parameters accept a scalar (equality) or a list (membership).
        Filters are AND-ed together.
        """
        df = self._df

        def _apply(col: str, val: Any) -> None:
            nonlocal df
            if val is None or col not in df.columns:
                return
            if isinstance(val, (list, tuple)):
                df = df[df[col].isin(val)]
            elif isinstance(val, str):
                df = df[df[col].astype(str).str.contains(val, case=False, regex=False, na=False)]
            else:
                df = df[df[col] == val]

        _apply("n_days", n_days)
        _apply("n_bins", n_bins)
        _apply("area", area)
        _apply("waste_type", waste_type)
        _apply("data_distribution", data_distribution)
        _apply("run_name", run_name)
        _apply("policy_slug", policy_slug)
        _apply("mandatory_selection", mandatory_selection)
        _apply("route_constructor", route_constructor)
        _apply("route_constructor_engine", route_constructor_engine)
        _apply("acceptance_criterion", acceptance_criterion)
        _apply("route_improver", route_improver)

        return ResultsDB(df.reset_index(drop=True))

    # ── Export ────────────────────────────────────────────────────────────────

    def to_dataframe(self, include_raw: bool = False) -> Any:
        """Return the underlying DataFrame.

        Args:
            include_raw: If True, keep the heavy ``samples`` and ``daily``
                columns (loaded only when requested in :func:`load_log_dict`).
        """
        df = self._df.copy()
        if not include_raw:
            df = df.drop(columns=[c for c in ("samples", "daily") if c in df.columns])
        return df

    @property
    def metric_cols(self) -> List[str]:
        """Mean metric column names (excludes metadata and ``_std`` columns)."""
        exclude = set(self.METADATA_COLS) | {"samples", "daily"}
        return [c for c in self._df.columns if c not in exclude and not c.endswith("_std")]

    @property
    def std_cols(self) -> List[str]:
        """Standard-deviation column names (columns ending with ``_std``)."""
        return [c for c in self._df.columns if c.endswith("_std")]

    def mean_metrics(self) -> Any:
        """Return a DataFrame with metadata + mean metric columns only."""
        cols = [c for c in self.METADATA_COLS if c in self._df.columns] + self.metric_cols
        return self._df[cols].copy()

    def std_metrics(self) -> Any:
        """Return a DataFrame with metadata + standard-deviation columns only."""
        cols = [c for c in self.METADATA_COLS if c in self._df.columns] + self.std_cols
        return self._df[cols].copy()

    def groupby(self, by: Any) -> Any:
        """Delegate to ``DataFrame.groupby`` for aggregation and pivoting."""
        return self._df.groupby(by)

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        def _uniq(col: str) -> Union[List[Any], str]:
            return sorted(self._df[col].dropna().unique().tolist()) if col in self._df.columns else "—"

        return (
            f"ResultsDB({len(self._df)} results)\n"
            f"  distributions : {_uniq('data_distribution')}\n"
            f"  n_bins        : {_uniq('n_bins')}\n"
            f"  run_names     : {_uniq('run_name')}\n"
            f"  constructors  : {_uniq('route_constructor')}"
        )


def load_log_dict(  # noqa: C901
    home_dir: str,
    output_dir: str = "output",
    *,
    mandatory_prefixes: Optional[List[str]] = None,
    route_constructors: Optional[List[str]] = None,
    route_improvers: Optional[List[str]] = None,
    route_constructor_engines: Optional[List[str]] = None,
    acceptance_criteria: Optional[List[str]] = None,
    load_samples: bool = False,
    load_daily: bool = False,
    lock: Optional[threading.Lock] = None,
) -> ResultsDB:
    """Walk all simulation output directories and return a queryable ResultsDB.

    Recursively scans ``{home_dir}/assets/{output_dir}/`` for files matching
    ``log_*_<N>N.json`` (realtime JSONL files are skipped).  Each file is
    parsed into a record containing:

    * **Path metadata**: ``n_days``, ``n_bins``, ``area``, ``waste_type``,
      ``data_distribution``, ``run_name``.
    * **Policy components** (best-effort slug decomposition):
      ``mandatory_selection``, ``route_constructor``, ``acceptance_criterion``,
      ``route_improver``, ``policy_slug``.
    * **Mean / std metrics**: read directly from the ``"mean"`` / ``"std"``
      sections of each JSON; computed from ``"samples"`` when those sections are
      absent.

    The expected path structure is::

        {home_dir}/assets/{output_dir}/
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
            route_constructors=["vrpp_gurobi", "gurobi", "my_solver"],
            route_improvers=["tsp_fast_tsp", "ftsp", "tsp"],
        )

    Args:
        home_dir: Project root (where ``assets/`` lives).
        output_dir: Sub-directory under ``assets/`` holding the results tree.
        mandatory_prefixes: Override the default ``_MANDATORY_PREFIXES`` list
            used by slug parsing.  Pass ``None`` to keep the built-in defaults.
        route_constructors: Override the default ``_ROUTE_CONSTRUCTORS`` list.
            Pass ``None`` to keep the built-in defaults.
        route_improvers: Override the default ``_ROUTE_IMPROVERS`` list.
            Pass ``None`` to keep the built-in defaults.
        route_constructor_engines: Override the default
            ``_ROUTE_CONSTRUCTOR_ENGINES`` list (empty by default — must be
            supplied when constructor engines appear in slug names).
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
        sub = db.filter(n_bins=100, mandatory_selection="lookahead",
                        data_distribution="emp")
        print(sub.mean_metrics())
        # Compare profit across constructors
        sub.groupby("route_constructor")["profit"].mean().sort_values()
    """
    import glob as _glob

    base_dir = os.path.join(home_dir, "assets", output_dir)
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
                route_constructors,
                route_improvers,
                route_constructor_engines,
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


def final_simulation_summary(log: Dict[str, Any], policy: str, n_samples: int) -> None:
    """Log a final summary of simulation statistics for a policy.

    Args:
        log: The summary log dictionary containing policy keys.
        policy: The specific policy name to summarize.
        n_samples: Total number of samples involved in the run.
    """
    if policy not in log:
        logger.warning(f"Policy {policy} not found in log for summary.")
        return

    # Use the pretty table for a single policy if that's all we have,
    # or if we want to maintain the single-policy logging behavior.
    display_simulation_summary_table(
        {policy: log[policy]},
        title=f"Simulation Summary: [bold cyan]{policy}[/] ({n_samples} samples)",
    )


def display_simulation_summary_table(  # noqa: C901
    log: Dict[str, Any],
    title: str = "Simulation Summary",
    lock: Optional[Any] = None,
) -> None:
    """Display a pretty comparative table of simulation results for multiple policies.

    Args:
        log: Dictionary mapping policy names to metric dictionaries or lists.
        title: Title for the table. Defaults to "Simulation Summary".
        lock: Optional lock for thread-safe printing. Defaults to None.
    """
    if not log:
        return

    console = Console()
    table = Table(
        title=title,
        box=box.DOUBLE_EDGE,
        header_style="bold magenta",
        border_style="blue",
        title_style="bold white on blue",
        show_header=True,
        expand=False,
    )

    # Core metrics to display in the table
    has_time_spent = any(
        isinstance(stats, dict) and "time_spent" in stats and stats["time_spent"] > 0
        for stats in log.values()
    )
    display_metrics = [
        ("Profit", "profit"),
        ("Collected", "kg"),
        ("Lost", "kg_lost"),
        ("Col", "ncol"),
        ("Dist", "km"),
        ("Eff", "kg/km"),
        ("Over", "overflows"),
        ("Days", "days"),
        ("Time", "time"),
    ]
    if has_time_spent:
        display_metrics.insert(-1, ("ShiftTime", "time_spent"))

    table.add_column("Policy", style="cyan", no_wrap=True)
    for label, _ in display_metrics:
        table.add_column(label, justify="right")

    for pol, stats in log.items():
        row = [pol]
        for _, key in display_metrics:
            if isinstance(stats, dict):
                val = stats.get(key, 0.0)
            elif isinstance(stats, (list, tuple)):
                try:
                    idx = udef.SIM_METRICS.index(key)
                    val = stats[idx]
                except (ValueError, IndexError):
                    val = 0.0
            else:
                val = 0.0

            if key == "profit":
                row.append(f"${val:,.2f}")
            elif key in ["kg", "kg_lost"]:
                row.append(f"{val:,.1f}kg")
            elif key == "km":
                row.append(f"{val:,.1f}km")
            elif key == "kg/km":
                row.append(f"{val:.2f}")
            elif key in ["overflows", "ncol", "days"]:
                color = "red" if key == "overflows" and val > 0 else "white"
                if key == "days":
                    color = "cyan"
                row.append(f"[{color}]{int(val)}[/]")
            elif key == "time":
                row.append(f"{val:.2f}s")
            elif key == "time_spent":
                row.append(f"{val:.2f}h")
            else:
                row.append(f"{val:.2f}")
        table.add_row(*row)

    if lock:
        with lock:
            console.print(table)
    else:
        console.print(table)


def display_per_policy_simulation_summary(  # noqa: C901
    pol_name: str,
    sample_id: int,
    aggregate_metrics: List[float],
    daily_log: Dict[str, List[Any]],
    title_prefix: str = "Results for",
    lock: Optional[Any] = None,
) -> None:
    """Display detailed results for a single policy simulation run.

    Args:
        pol_name: Name of the policy.
        sample_id: ID of the sample/seed.
        aggregate_metrics: List of aggregate metrics for the entire run.
        daily_log: Dictionary of daily metrics.
        title_prefix: Prefix for the table titles.
        lock: Optional lock for thread-safe printing.
    """
    console = Console()

    # 1. STATISTICS SUMMARY TABLE
    summary_title = f"{title_prefix} [bold cyan]{pol_name}[/] (Sample #{sample_id}) - [yellow]Stats Summary[/]"

    def _print_tables():  # noqa: C901
        try:
            # Using rule for better separation
            console.rule(f"[bold white on blue] {summary_title} [/]")
            display_simulation_summary_table({pol_name: aggregate_metrics}, title=None, lock=None)

            # 2. DAILY PERFORMANCE TABLE (Filtered)
            daily_title = f"{title_prefix} [bold cyan]{pol_name}[/] (Sample #{sample_id}) - [green]Daily Routes[/]"

            table = Table(
                box=box.MINIMAL_DOUBLE_HEAD,
                header_style="bold magenta",
                border_style="green",
                show_header=True,
                expand=False,
            )

            columns = [
                ("Day", "day", "cyan"),
                ("Mandatory", "mandatory_nodes", "yellow"),
                ("Tour", "tour", "white"),
                ("Profit", "profit", "green"),
                ("KG", "kg", "white"),
                ("Lost", "kg_lost", "red"),
                ("Col", "ncol", "white"),
                ("Dist", "km", "white"),
                ("Eff", "kg/km", "yellow"),
                ("Over", "overflows", "red"),
            ]
            if "time_spent" in daily_log and any(
                isinstance(v, (int, float)) and v > 0 for v in daily_log["time_spent"]
            ):
                columns.append(("ShiftTime", "time_spent", "white"))

            for label, _, style in columns:
                table.add_column(label, style=style, justify="right" if label not in ("Tour", "Mandatory") else "left")

            # Use km list length as the authoritative iteration count
            kms = daily_log.get("km", [])

            has_active_days = False
            for i, km_val in enumerate(kms):
                # Only show days where a route was performed (km > 0)
                if km_val > 0:
                    has_active_days = True
                    row = []
                    for _, key, _ in columns:
                        if key == "day" or key is None:
                            # "Day" column — synthetic 1-based index (not stored in daily_log)
                            row.append(str(i + 1))
                            continue
                        vals = daily_log.get(key, [])
                        val = vals[i] if i < len(vals) else None

                        if key == "mandatory_nodes":
                            mand_str = str(val) if val else "[]"
                            if len(mand_str) > 40:
                                mand_str = mand_str[:37] + "..."
                            row.append(mand_str)
                        elif key == "tour":
                            tour_str = str(val) if val is not None else "[]"
                            if len(tour_str) > 50:
                                tour_str = tour_str[:47] + "..."
                            row.append(tour_str)
                        elif key == "profit":
                            if val is not None:
                                color = "green" if val > 0 else "red"
                                row.append(f"[{color}]${val:,.2f}[/]")
                            else:
                                row.append("-")
                        elif key in ["kg", "kg_lost"]:
                            row.append(f"{val:,.1f}kg" if val is not None else "0.0kg")
                        elif key == "km":
                            row.append(f"{val:,.1f}km" if val is not None else "0.0km")
                        elif key == "kg/km":
                            row.append(f"{val:.2f}" if val is not None else "0.00")
                        elif key == "overflows":
                            if val is not None:
                                color = "red" if val > 0 else "white"
                                row.append(f"[{color}]{int(val)}[/]")
                            else:
                                row.append("0")
                        elif key == "ncol":
                            row.append(f"{int(val)}" if val is not None else "0")
                        elif key == "time_spent":
                            row.append(f"{val:.2f}h" if val is not None else "0.00h")
                        else:
                            row.append(str(val))
                    table.add_row(*row)

            if has_active_days:
                console.print("\n")
                console.rule(f"[bold white on green] {daily_title} [/]")
                console.print(table)
            else:
                console.print("\n[yellow]No routes performed during this simulation run (all KM=0).[/]")

            console.print("\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to display simulation summary: {e}")
            traceback.print_exc()

    if lock:
        with lock:
            _print_tables()
    else:
        _print_tables()
