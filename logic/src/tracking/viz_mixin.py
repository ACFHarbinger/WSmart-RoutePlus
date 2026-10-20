"""Policy Visualization Mixin.

Provides lightweight, zero-overhead state recording for classical and
heuristic policies. Import :class:`PolicyVizMixin` and add it as a
mixin base-class to any policy that should expose iteration data to the
Streamlit dashboard.

Attributes:
    PolicyStateRecorder: Fixed-capacity ring-buffer for per-iteration policy telemetry.
    PolicyVizMixin: Mixin that adds zero-overhead telemetry to policy classes.

Example:
    >>> class VectorizedALNS(PolicyVizMixin):
    ...     def solve(self, *args, **kwargs):
    ...         for i in range(10):
    ...             self._viz_record(iteration=i, best_cost=100.0 - i)
    >>> solver = VectorizedALNS()
    >>> solver.solve()
    >>> data = solver.get_viz_data()
    >>> data["iteration"]
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
"""

from typing import Any, Dict, List


class PolicyStateRecorder:
    """Fixed-capacity ring-buffer for per-iteration policy telemetry.

    Stores a history of scalar metrics as lists of values, enabling time-series
    visualization of optimization progress.

    Attributes:
        _buf: Dictionary mapping metric keys to lists of recorded values.
        _max: Maximum number of records per metric key to keep in memory.
    """

    def __init__(self, max_history: int = 5000) -> None:
        """Initializes the PolicyStateRecorder.

        Args:
            max_history: Maximum number of records to keep. Oldest entries
                are silently dropped once the buffer is full. Defaults to 5000.
        """
        self._buf: Dict[str, List[Any]] = {}
        self._max = max_history

    def record(self, **kwargs: Any) -> None:
        """Appends one snapshot of keyword-named scalar metrics.

        Args:
            kwargs: Arbitrary scalar metrics to record (e.g., best_cost=1.2).
        """
        for key, val in kwargs.items():
            lst = self._buf.setdefault(key, [])
            lst.append(val)
            if len(lst) > self._max:
                lst.pop(0)

    def get(self) -> Dict[str, List[Any]]:
        """Retrieves a shallow copy of the accumulated telemetry.

        Returns:
            Dict[str, List[Any]]: Mapping of metric names to lists of history.
        """
        return {k: list(v) for k, v in self._buf.items()}

    def reset(self) -> None:
        """Discards all recorded data from the buffer."""
        self._buf.clear()

    def __len__(self) -> int:
        """Returns the number of records in the longest metric history.

        Returns:
            int: The maximum length among all recorded metric lists.
        """
        if not self._buf:
            return 0
        return max(len(v) for v in self._buf.values())


class PolicyVizMixin:
    """Mixin that adds zero-overhead telemetry to policy classes.

    The internal PolicyStateRecorder is only created on the first call to
    _viz_record, so policies that never record data pay no allocation cost.
    Concrete policies should call self._viz_record(**metrics) inside their
    main iteration loop.

    Attributes:
        _viz_recorder: The lazily-initialized recorder instance.
    """

    @property
    def _viz(self) -> PolicyStateRecorder:
        """Lazy property that initializes and returns the state recorder.

        Returns:
            PolicyStateRecorder: The process-private telemetry recorder.
        """
        try:
            return self.__viz_recorder  # type: ignore[attr-defined]
        except AttributeError:
            self.__viz_recorder: PolicyStateRecorder = PolicyStateRecorder()
            return self.__viz_recorder

    def _viz_record(self, **kwargs: Any) -> None:
        """Records one snapshot of metrics. Call once per iteration.

        Args:
            kwargs: Keyword arguments representing the metrics to record.
        """
        self._viz.record(**kwargs)

    def get_viz_data(self) -> Dict[str, List[Any]]:
        """Retrieves accumulated telemetry.

        Returns:
            Dict[str, List[Any]]: Accumulated telemetry as a plain dict of
                lists, or an empty dict if _viz_record was never called.
        """
        try:
            return self.__viz_recorder.get()  # type: ignore[attr-defined]
        except AttributeError:
            return {}
