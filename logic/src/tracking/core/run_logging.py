"""Tag, param and metric logging for tracking Run.

Moved from ``logic.src.tracking.core.run``.
"""

from __future__ import annotations

import contextlib
import math
import threading
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, TypeVar

if TYPE_CHECKING:
    from logic.src.tracking.core.store import TrackingStore

_RunT = TypeVar("_RunT", bound="RunLoggingMixin")


class RunLoggingMixin:
    """Moved methods of :class:`Run`."""

    run_id: str
    _store: "TrackingStore"
    _closed: bool
    _sinks: List[Any]
    _buffer_lock: threading.Lock
    _metric_buffer: List[Tuple[str, float, int]]
    _buffer_size: int

    def add_sink(self: _RunT, sink: Any) -> _RunT:
        """Attaches a secondary logging sink to this run.

        The sink receives forwarded calls for every metric, param, and
        artifact logged. Sinks are useful for mirroring data to external
        services like MLflow or W&B.

        Args:
            sink: Object implementing log_metric, log_params, log_artifact,
                and finish methods.

        Returns:
            Run: Self for method chaining.
        """
        self._sinks.append(sink)
        return self

    def set_tag(self: _RunT, key: str, value: Any) -> _RunT:
        """Attaches a metadata tag to this run.

        Args:
            key: Semantic name for the tag.
            value: Tag value (will be cast to string).

        Returns:
            Run: Self for method chaining.
        """
        if not self._closed:
            self._store.set_tag(self.run_id, key, str(value))
        return self

    def set_tags(self: _RunT, tags: Dict[str, Any]) -> _RunT:
        """Attaches multiple metadata tags to this run.

        Args:
            tags: Dictionary of tag keys and values.

        Returns:
            Run: Self for method chaining.
        """
        if not self._closed:
            self._store.set_tags(self.run_id, {k: str(v) for k, v in tags.items()})
        return self

    def log_param(self: _RunT, key: str, value: Any) -> _RunT:
        """Records a single configuration parameter.

        Args:
            key: Name of the parameter.
            value: Parameter value.

        Returns:
            Run: Self for method chaining.
        """
        if not self._closed:
            self._store.log_param(self.run_id, key, value)
        return self

    def log_params(self: _RunT, params: Dict[str, Any]) -> _RunT:
        """Records multiple configuration parameters, flattening nested dicts.

        Args:
            params: Dictionary of parameters to log.

        Returns:
            Run: Self for method chaining.
        """
        if not self._closed:
            flat = _flatten_dict(params)
            self._store.log_params(self.run_id, flat)
            for sink in self._sinks:
                with contextlib.suppress(Exception):
                    sink.log_params(flat)
        return self

    def log_metric(self: _RunT, key: str, value: Any, step: int = 0) -> _RunT:
        """Logs a single scalar metric value.

        Args:
            key: Name of the metric.
            value: Scalar value to log (must be convertible to float).
            step: Training or simulation step index.

        Returns:
            Run: Self for method chaining.
        """
        if self._closed:
            return self
        fval = _safe_float(value)
        if fval is None:
            return self
        with self._buffer_lock:
            self._metric_buffer.append((key, fval, step))
            if len(self._metric_buffer) >= self._buffer_size:
                self._flush_metrics_locked()
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                sink.log_metric(key, fval, step)
        return self

    def log_metrics(self: _RunT, metrics: Dict[str, Any], step: int = 0) -> _RunT:
        """Logs a dictionary of scalar metric values.

        Args:
            metrics: Mapping of metric keys to values.
            step: Training or simulation step index.

        Returns:
            Run: Self for method chaining.
        """
        if self._closed:
            return self
        with self._buffer_lock:
            for k, v in metrics.items():
                fval = _safe_float(v)
                if fval is not None:
                    self._metric_buffer.append((k, fval, step))
            if len(self._metric_buffer) >= self._buffer_size:
                self._flush_metrics_locked()
        return self

    def flush(self: _RunT) -> _RunT:
        """Force-flushes any buffered metrics to the storage backend.

        Returns:
            Run: Self for method chaining.
        """
        with self._buffer_lock:
            self._flush_metrics_locked()
        return self

    def _flush_metrics_locked(self: _RunT) -> None:
        """Internal helper to flush the metric buffer.

        Caller must hold `_buffer_lock`.
        """
        if self._metric_buffer:
            self._store.log_metrics_batch(self.run_id, self._metric_buffer)
            self._metric_buffer.clear()


def _flatten_dict(d: Dict[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flattens a nested dictionary using separator-joined keys.

    Args:
        d: The nested dictionary to flatten.
        prefix: Current key prefix for recursive calls.
        sep: Separator character between nested levels.

    Returns:
        Dict[str, Any]: A single-level dictionary with flattened keys.
    """
    result: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, key, sep))
        else:
            result[key] = v
    return result


def _safe_float(v: Any) -> Optional[float]:
    """Safely converts a value to a float, returning None on failure or overflow.

    Args:
        v: The value to convert.

    Returns:
        Optional[float]: The converted float value, or None if invalid.
    """
    try:
        fval = float(v)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return fval
    except (TypeError, ValueError):
        return None
