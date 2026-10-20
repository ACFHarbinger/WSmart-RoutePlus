"""Queryable ResultsDB container for simulation logs.

Moved from ``logic.src.tracking.logging.modules.analysis``.
"""

from typing import Any, Dict, List, Optional, Union


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
        df = db.filter(packing="lookahead", n_bins=100).mean_metrics()
        # Group by routing algorithm, average profit across graph sizes
        db.to_dataframe().groupby("routing_algorithm")["profit"].mean()
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
        "packing",
        "routing_algorithm",
        "routing_algorithm_engine",
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
        packing: Optional[str] = None,
        routing_algorithm: Optional[str] = None,
        routing_algorithm_engine: Optional[str] = None,
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
        _apply("packing", packing)
        _apply("routing_algorithm", routing_algorithm)
        _apply("routing_algorithm_engine", routing_algorithm_engine)
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
            f"  constructors  : {_uniq('routing_algorithm')}"
        )
