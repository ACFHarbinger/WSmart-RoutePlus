"""In-process fakes for the cloud runner's interfaces.

Attributes:
    FakeClock: A settable clock.
    MemoryObjectStore: :class:`~logic.controllers.cloud.interfaces.ObjectStore` in memory.
    ScriptedBackend: A backend whose attempts follow a script (succeed, fail, preempt, hang).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from logic.controllers.cloud.interfaces import (
    AttemptState,
    BackendCapabilities,
    BackendStatus,
    CapacityError,
    ComputeBackend,
    LaunchSpec,
    ObjectStore,
)
from logic.controllers.cloud.model import ComputeKind, Provider, ResultFile


class FakeClock:
    """A clock that only moves when told to."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class MemoryObjectStore(ObjectStore):
    """Objects in a dict."""

    def __init__(self) -> None:
        self.objects: Dict[str, bytes] = {}

    def put_file(self, local: Path, key: str) -> ResultFile:
        data = Path(local).read_bytes()
        self.objects[key] = data
        return ResultFile(relpath=key, key=key, sha256=hashlib.sha256(data).hexdigest(), size=len(data))

    def get_file(self, key: str, local: Path) -> None:
        if key not in self.objects:
            raise KeyError(key)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(self.objects[key])

    def exists(self, key: str) -> bool:
        return key in self.objects

    def list(self, prefix: str) -> List[str]:
        return sorted(key for key in self.objects if key.startswith(prefix))

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)


@dataclass
class _Attempt:
    launch: LaunchSpec
    outcome: str
    polls: int = 0
    cancelled: bool = False
    log: List[str] = field(default_factory=list)


class ScriptedBackend(ComputeBackend):
    """A backend whose attempts follow a script.

    Args:
        script: ``(task) -> outcome`` per attempt, where outcome is ``"succeed"``,
            ``"fail"``, ``"preempt"``, ``"hang"`` or ``"no_capacity"``. Default: succeed.
        polls_to_finish: Polls an attempt spends running before its outcome.
        on_success: Called with the launch when an attempt succeeds (to write fake outputs).
        provider: Provider to report.
        kind: Compute kind to report.
    """

    def __init__(
        self,
        script: Optional[Callable[[LaunchSpec], str]] = None,
        polls_to_finish: int = 1,
        on_success: Optional[Callable[[LaunchSpec], None]] = None,
        provider: Provider = Provider.LOCAL,
        kind: ComputeKind = ComputeKind.PROCESS,
    ) -> None:
        self.script = script or (lambda launch: "succeed")
        self.polls_to_finish = polls_to_finish
        self.on_success = on_success
        self.provider = provider
        self.kind = kind
        self.attempts: Dict[str, _Attempt] = {}

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(provider=self.provider, kinds=[self.kind], supports_spot=True)

    def submit(self, launch: LaunchSpec) -> str:
        outcome = self.script(launch)
        if outcome == "no_capacity":
            raise CapacityError("scripted: no capacity")
        handle = f"fake-{len(self.attempts) + 1}"
        self.attempts[handle] = _Attempt(launch=launch, outcome=outcome, log=[f"start {launch.task.task_id}"])
        return handle

    def poll(self, handle: str) -> BackendStatus:
        attempt = self.attempts.get(handle)
        if attempt is None:
            return BackendStatus(AttemptState.UNKNOWN, message="no such attempt")
        if attempt.cancelled:
            return BackendStatus(AttemptState.FAILED, exit_code=143, message="cancelled")
        attempt.polls += 1
        if attempt.outcome == "hang" or attempt.polls <= self.polls_to_finish:
            return BackendStatus(AttemptState.RUNNING)
        if attempt.outcome == "succeed":
            if self.on_success is not None and attempt.polls == self.polls_to_finish + 1:
                self.on_success(attempt.launch)
            return BackendStatus(AttemptState.SUCCEEDED, exit_code=0)
        if attempt.outcome == "preempt":
            return BackendStatus(AttemptState.PREEMPTED, message="scripted preemption")
        return BackendStatus(AttemptState.FAILED, exit_code=1, message="scripted failure")

    def cancel(self, handle: str) -> None:
        if handle in self.attempts:
            self.attempts[handle].cancelled = True

    def logs(self, handle: str, tail: Optional[int] = None) -> str:
        lines = self.attempts[handle].log if handle in self.attempts else []
        return "\n".join(lines if tail is None else lines[-tail:])
