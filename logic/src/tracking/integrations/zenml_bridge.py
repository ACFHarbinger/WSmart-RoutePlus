"""ZenML secondary sink and stack helpers for WSTracker runs.

This module provides the :class:`ZenMLBridge` which enables integration between
WSTracker and the ZenML environment. It mirrors experiment data to a
ZenML-managed MLflow experiment tracker. It also provides utility functions
for programmatic ZenML stack configuration.

Attributes:
    ZenMLBridge: A sink that forwards experiment data to ZenML-managed MLflow.

Example:
    >>> @step(experiment_tracker="mlflow_tracker")
    ... def train_step():
    ...     bridge = ZenMLBridge.attach(run)
    ...     # Metrics now flow through WSTracker -> ZenML -> MLflow
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict

from logic.src.tracking.core.run import Run
from logic.src.tracking.logging.pylogger import get_pylogger

logger = get_pylogger(__name__)

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment,misc]

try:
    from zenml.client import Client
    from zenml.enums import StackComponentType
except ImportError:
    Client = None  # type: ignore[assignment,misc]
    StackComponentType = None  # type: ignore[assignment,misc]


class ZenMLBridge:
    """Forwards :class:`Run` events to ZenML-managed MLflow.

    This sink assumes that ZenML has already initialized an MLflow run for the
    current execution context. It acts as a passive proxy, delegating lifecycle
    management (start/stop) to the ZenML framework while mirroring actual
    experiment data points.

    Attributes:
        _mlflow: Cached reference to the mlflow module.
    """

    def __init__(self) -> None:
        """Initializes the ZenML bridge."""
        self._mlflow = mlflow

    # ------------------------------------------------------------------
    # Internal guard
    # ------------------------------------------------------------------

    def _has_active_run(self) -> bool:
        """Determines if a ZenML-owned MLflow run is currently active.

        Returns:
            bool: True if metrics can be forwarded to an active MLflow run.
        """
        if self._mlflow is None:
            return False
        with contextlib.suppress(Exception):
            return self._mlflow.active_run() is not None
        return False

    # ------------------------------------------------------------------
    # Sink protocol
    # ------------------------------------------------------------------

    def log_metric(self, key: str, value: float, step: int) -> None:
        """Forwards a scalar metric to the active ZenML experiment tracker.

        Args:
            key: Name of the metric.
            value: Scalar value.
            step: Training or evaluation step index.
        """
        if not self._has_active_run():
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_metric(key, value, step=step)  # type: ignore[union-attr]

    def log_params(self, params: Dict[str, Any]) -> None:
        """Forwards multiple parameters to the active ZenML experiment tracker.

        Args:
            params: Mapping of parameter names to values (coerced to strings).
        """
        if not self._has_active_run():
            return
        with contextlib.suppress(Exception):
            str_params = {k: str(v) for k, v in params.items()}
            items = list(str_params.items())
            for i in range(0, len(items), 100):
                self._mlflow.log_params(dict(items[i : i + 100]))  # type: ignore[union-attr]

    def log_artifact(self, path: str) -> None:
        """Forwards a local file artifact to the ZenML-managed MLflow storage.

        Args:
            path: Absolute path to the file to log.
        """
        if not self._has_active_run():
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_artifact(path)  # type: ignore[union-attr]

    def finish(self, status: str = "completed") -> None:
        """Terminates tracking. (No-op as ZenML manages the run lifecycle).

        Args:
            status: Final status of the WSTracker run.
        """
        # Intentionally empty — calling mlflow.end_run() here would terminate
        # ZenML's experiment tracking run prematurely.

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def attach(cls, run: Run) -> "ZenMLBridge":
        """Factory method to create and attach a bridge to an existing run.

        Args:
            run: The WSTracker run instance to mirror.

        Returns:
            ZenMLBridge: The initialized and attached bridge instance.
        """
        bridge = cls()
        run.add_sink(bridge)
        return bridge


# ---------------------------------------------------------------------------
# ZenML stack helpers
# ---------------------------------------------------------------------------


def configure_zenml_stack(
    mlflow_tracking_uri: str,
    stack_name: str = "wsmart-route-stack",
) -> bool:
    """Configures a local ZenML stack with MLflow tracking.

    Programmatically initializes a stack with an associated MLflow experiment
    tracker. If the components already exist, they are activated for use.

    Args:
        mlflow_tracking_uri: URI for the MLflow tracking server.
        stack_name: Unique name for the ZenML stack.

    Returns:
        bool: True if configuration and activation were successful.
    """
    if Client is None or StackComponentType is None:
        logger.warning("ZenML not installed; cannot configure stack.")
        return False

    try:
        client = Client()
        tracker_component_name = f"mlflow-tracker-{stack_name}"

        # Register the MLflow experiment tracker component
        with contextlib.suppress(Exception):
            client.create_stack_component(
                name=tracker_component_name,
                # Use the Enum member instead of the literal string
                component_type=StackComponentType.EXPERIMENT_TRACKER,
                flavor="mlflow",
                configuration={"tracking_uri": mlflow_tracking_uri},
            )

        # Create the stack
        with contextlib.suppress(Exception):
            client.create_stack(
                name=stack_name,
                components={StackComponentType.EXPERIMENT_TRACKER: tracker_component_name},
            )

        client.activate_stack(stack_name)
        return True

    except Exception as exc:
        logger.warning(f"ZenML stack configuration failed (non-fatal): {exc}")
        return False
