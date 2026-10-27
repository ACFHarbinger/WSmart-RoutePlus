"""Internal logging helper for processor events.

Attributes:
    _log_processor_event: Logs a data processing event to the active tracking run (if any).

Example:
    >>> from logic.src.data.processor._logging import _log_processor_event
    >>> _log_processor_event("test_event", shape=(10,))
    >>>
"""

import contextlib
import sys

try:
    from logic.src.tracking.core.run import (  # pyrefly: ignore[missing-import]  # tracking is excluded from the minimal package
        get_active_run,
    )
except ImportError:
    get_active_run = None  # type: ignore[assignment]


def _log_processor_event(event_name, variable_name="data", event_type="mutate", shape=None, **kwargs):
    """Log a data processing event to the active tracking run (if any).

    Args:
        event_name: Description of event_name.
        variable_name: Description of variable_name.
        event_type: Description of event_type.
        shape: Description of shape.
        kwargs: Description of kwargs.
    """
    try:
        source_line = sys._getframe(1).f_lineno
    except Exception:
        source_line = 0

    with contextlib.suppress(Exception):
        run = get_active_run() if get_active_run is not None else None
        if run is not None:
            metadata = {
                "event": event_name,
                "variable_name": variable_name,
                "source_file": "data/processor",
                "source_line": source_line,
            }
            metadata.update(kwargs)
            safe_meta = {}
            for k, v in metadata.items():
                if isinstance(v, (int, float, str, bool, type(None))):
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)
            run.log_dataset_event(event_type, shape=shape, metadata=safe_meta)
