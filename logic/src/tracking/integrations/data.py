"""Runtime data mutation tracking for WSTracker.

This module provides the :class:`RuntimeDataTracker` which attaches to a
tracking run and records statistical snapshots of tensor datasets as they
are loaded, regenerated, or mutated in memory. It captures distribution
metrics (mean, std, min, max) to detect dataset drift over training epochs.

Attributes:
    RuntimeDataTracker: Tracks in-memory dataset distribution mutations.

Example:
    >>> from logic.src.tracking.integrations.data import RuntimeDataTracker
    >>> tracker = RuntimeDataTracker(run)
    >>> tracker.on_load(train_dataset.tensor_dict)
"""

from __future__ import annotations

import contextlib
import itertools
import sys
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np


def _is_array(value: Any) -> bool:
    torch = sys.modules.get("torch")
    return isinstance(value, np.ndarray) or (torch is not None and isinstance(value, torch.Tensor))


if TYPE_CHECKING:
    from logic.src.tracking.core.run import Run


class RuntimeDataTracker:
    """Tracks in-memory dataset mutations during a training or simulation run.

    Records per-field tensor statistics as run parameters on every snapshot,
    enabling visibility into distribution shifts across epochs. Parameters are
    namespaced as ``data/{tag}/{field}/{stat}``.

    Attributes:
        _run: The active tracking run instance.
        _max_fields: Maximum number of fields to snapshot per call.
        _history: Chronological list of computed statistics snapshots.
    """

    def __init__(self, run: Optional[Run], max_fields: int = 32) -> None:
        """Initializes the data mutation tracker.

        Args:
            run: The active tracking run. If None, operations become no-ops.
            max_fields: Maximum number of tensor fields to snapshot per call.
                Defaults to 32.
        """
        self._run = run
        self._max_fields = max_fields
        self._history: List[Tuple[str, Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def snapshot(
        self,
        data: Any,
        tag: str,
        step: Optional[int] = None,
    ) -> None:
        """Captures tensor statistics from data and logs them to the run.

        Args:
            data: The in-memory dataset payload (Tensor, dict, or object with
                .data attribute).
            tag: Namespace prefix for parameter keys (e.g., 'val/epoch_5').
            step: Optional step index for logging per-field mean as a metric.
        """
        if self._run is None:
            return
        fields = self._extract_fields(data)
        stats = self._compute_stats(fields)
        self._log_stats(stats, tag, step)
        self._history.append((tag, stats))

    def on_load(
        self,
        data: Any,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_event: bool = True,
    ) -> None:
        """Snapshots a dataset variable immediately after initial loading.

        Args:
            data: The in-memory dataset object.
            shape: Explicit logical shape; inferred from data if omitted.
            metadata: Extra context for the load event.
            log_event: If True, registers a 'load' event in the run history.
        """
        if self._run is None:
            return
        n = shape or self._infer_shape(data)
        if log_event:
            meta: Dict[str, Any] = {
                "event": "load",
                "shape": n,
            }
            if metadata:
                meta.update(metadata)
            self._run.log_dataset_event("load", shape=n, metadata=meta)
        self.snapshot(data, tag="load")

    def on_regenerate(
        self,
        data: Any,
        epoch: int,
        shape: Optional[tuple] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_event: bool = True,
    ) -> None:
        """Records a dataset mutation event for a new training epoch.

        Args:
            data: The freshly regenerated dataset variable.
            epoch: Current training epoch index.
            shape: Explicit logical shape; inferred from data if omitted.
            metadata: Extra context for the regeneration event.
            log_event: If True, registers a 'mutate' event in the run history.
        """
        if self._run is None:
            return
        n = shape or self._infer_shape(data)
        if log_event:
            meta: Dict[str, Any] = {
                "event": "regenerate",
                "epoch": epoch,
                "shape": n,
            }
            if metadata:
                meta.update(metadata)
            self._run.log_dataset_event("mutate", shape=n, metadata=meta)
        self.snapshot(data, tag=f"train/epoch_{epoch}", step=epoch)

    def on_augment(
        self,
        data: Any,
        description: str,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_event: bool = True,
    ) -> None:
        """Records a transformation applied to the in-memory dataset.

        Args:
            data: The dataset after the transform.
            description: Label for the transformation (e.g., 'rotation').
            step: Optional metric step index.
            metadata: Extra context for the augmentation event.
            log_event: If True, registers a 'mutate' event in the run history.
        """
        if self._run is None:
            return
        if log_event:
            meta: Dict[str, Any] = {
                "event": "augment",
                "description": description,
            }
            if metadata:
                meta.update(metadata)
            self._run.log_dataset_event("mutate", metadata=meta)
        self.snapshot(data, tag=f"augment/{description}", step=step)

    # ------------------------------------------------------------------
    # Delta / drift helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_fields(self, data: Any) -> Dict[str, Any]:
        """Unwraps data into a mapping of field names to torch tensors.

        Args:
            data: The payload to extract tensors from.

        Returns:
            Dict[str, Any]: Mapping of name to tensor.
        """
        if not _is_array(data) and hasattr(data, "data"):
            data = data.data

        if _is_array(data):
            return {"data": data}

        if isinstance(data, dict):
            return {k: v for k, v in list(data.items())[: self._max_fields] if _is_array(v)}

        # TensorDict-like (tensordict library or similar mapping)
        if hasattr(data, "keys") and callable(data.keys):
            result: Dict[str, Any] = {}
            for k in itertools.islice(cast(Iterable[Any], data.keys()), self._max_fields):
                with contextlib.suppress(Exception):
                    v = data[k]
                    if _is_array(v):
                        result[k] = v
            return result

        return {}

    @staticmethod
    def _compute_stats(fields: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Computes descriptive statistics for a dictionary of tensors.

        Args:
            fields: Mapping of name to tensor.

        Returns:
            Dict[str, Dict[str, Any]]: Per-field statistics.
        """
        stats: Dict[str, Dict[str, Any]] = {}
        for name, tensor in fields.items():
            if isinstance(tensor, np.ndarray):
                t = tensor.astype(np.float32, copy=False)
                entry: Dict[str, Any] = {"shape": list(t.shape), "numel": t.size}
                if t.size:
                    entry.update(
                        mean=round(float(t.mean()), 6),
                        std=round(float(t.std(ddof=1)), 6) if t.size > 1 else 0.0,
                        min=round(float(t.min()), 6),
                        max=round(float(t.max()), 6),
                    )
                stats[name] = entry
                continue
            t = tensor.float().detach()
            shape: List[int] = [d for d in t.shape]
            try:
                entry = {
                    "shape": shape,
                    "numel": t.numel(),
                    "mean": round(t.mean().item(), 6),
                    "std": round(t.std().item(), 6) if t.numel() > 1 else 0.0,
                    "min": round(t.min().item(), 6),
                    "max": round(t.max().item(), 6),
                }
            except Exception:
                entry = {"shape": shape, "numel": t.numel()}
            stats[name] = entry
        return stats

    def _log_stats(
        self,
        stats: Dict[str, Dict[str, Any]],
        tag: str,
        step: Optional[int],
    ) -> None:
        """Flattens and logs statistics as run parameters and metrics.

        Args:
            stats: Computed statistics dictionary.
            tag: Snapshot tag for namespacing.
            step: Optional step index for metrics.
        """
        params: Dict[str, Any] = {}
        for field, entry in stats.items():
            for stat_name, val in entry.items():
                params[f"data/{tag}/{field}/{stat_name}"] = val

        if params:
            assert self._run is not None
            self._run.log_params(params)

        if step is not None:
            assert self._run is not None
            for field, entry in stats.items():
                if "mean" in entry:
                    with contextlib.suppress(Exception):
                        self._run.log_metric(f"data/{field}/mean", float(entry["mean"]), step=step)

    @staticmethod
    def _infer_shape(data: Any) -> Optional[tuple]:
        """Provides a best-effort shape inference for a data object.

        Args:
            data: Object to inspect.

        Returns:
            Optional[tuple]: Dimension tuple or None.
        """
        if hasattr(data, "shape"):
            with contextlib.suppress(Exception):
                return tuple(data.shape)
        if hasattr(data, "data") and hasattr(data.data, "shape"):
            with contextlib.suppress(Exception):
                return tuple(data.data.shape)
        with contextlib.suppress(Exception):
            return (len(data),)
        return None
