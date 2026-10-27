"""Keep placing ownership alive during slow synchronous provider provisioning."""

import threading
from contextlib import contextmanager
from typing import Generator, cast

from logic.controllers.cloud.interfaces import CapacityError, ControlPlane
from logic.controllers.cloud.model import TaskRecord


@contextmanager
def renew_provisioning_lease(cp: ControlPlane, task: TaskRecord, seconds: float) -> Generator[None, None, None]:
    """Renew while submit blocks; a lost lease prevents successful scheduler handoff."""
    if not task.lease_owner:
        yield
        return
    stop = threading.Event()
    errors = []

    def renew() -> None:
        while not stop.wait(seconds / 3):
            try:
                if not cp.renew_lease(task.task_id, cast(str, task.lease_owner), seconds):
                    raise CapacityError("Provisioning lease no longer belongs to this scheduler")
            except Exception as exc:
                errors.append(exc)
                return

    thread = threading.Thread(target=renew, daemon=True, name="cloud-provisioning-lease")
    thread.start()
    try:
        yield
        if errors:
            raise CapacityError("Could not retain provisioning ownership") from errors[0]
    finally:
        stop.set()
        thread.join()
